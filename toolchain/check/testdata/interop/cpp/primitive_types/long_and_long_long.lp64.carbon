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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLongAndI128() {
  let x: Cpp.long_long = 1;
  let y: i128 = 1;
  AssertSameType(x + y, y);
  AssertSameType(y + x, y);
}

// --- bitwise_homogeneous_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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
// CHECK:STDOUT:   %cpp_long.patt: %pattern_type.f57 = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.227: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.235: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.227) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.235) [concrete]
// CHECK:STDOUT:   %.03f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da7, %ImplicitAs.facet.235 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c5e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d98: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_long.patt: %pattern_type.f57 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.998: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.f25(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0fe: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl.f25(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.1a1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.f25(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d9: %Core.IntLiteral.as.As.impl.Convert.type.1a1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.998 = facet_value Core.IntLiteral, (%As.impl_witness.0fe) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7f7: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %As.WithSelf.Convert.type.7f7, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d9, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.991: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d20: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.204, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.ad9: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.be5: %Int.as.ImplicitAs.impl.Convert.type.ad9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b6d: %ImplicitAs.type.7cb = facet_value %i64, (%ImplicitAs.impl_witness.d20) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.95a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.b6d) [concrete]
// CHECK:STDOUT:   %.f05: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.95a, %ImplicitAs.facet.b6d [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.d98, %Int.as.ImplicitAs.impl.Convert.be5 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.be5, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.e21: <bound method> = bound_method %int_1.d98, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2fa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fcd: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.fcd) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7cb, %ImplicitAs.facet.fcd [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.204 = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.f25.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.f25.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl.f25 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.03f = impl_witness_access constants.%ImplicitAs.impl_witness.227, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.c5e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_28.2(%int_1.loc8) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8_28.2: %i64 = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long: %i64 = wrapper_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.f57 = value_binding_pattern cpp_long [concrete = constants.%cpp_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = wrapper_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.f57 = value_binding_pattern carbon_long [concrete = constants.%carbon_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.7fe = impl_witness_access constants.%ImplicitAs.impl_witness.2fa, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
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
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.8fd = impl_witness_access constants.%As.impl_witness.0fe, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d9]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.991]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.f05 = impl_witness_access constants.%ImplicitAs.impl_witness.d20, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.be5]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.e21]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.3ec: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.3d3(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.06e: %UInt.as.ImplicitAs.impl.Convert.type.3ec = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8d9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f01, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.633: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e08: %ImplicitAs.type.14f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.8d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.17c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.e08) [concrete]
// CHECK:STDOUT:   %.ed5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.17c, %ImplicitAs.facet.e08 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.633 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.633, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.7e7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.01f: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.72f: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.fb3: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.e51(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.f86: %Core.IntLiteral.as.As.impl.Convert.type.fb3 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.5477: <witness> = impl_witness imports.%As.impl_witness_table.637, @Core.IntLiteral.as.As.impl.e51(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.49f: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.e51(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.990: %Core.IntLiteral.as.As.impl.Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.72f = facet_value Core.IntLiteral, (%As.impl_witness.5477) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e94: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.b3c: type = fn_type_with_self_type %As.WithSelf.Convert.type.e94, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.990 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.990, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.318: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.faa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.465, @UInt.as.ImplicitAs.impl.3d3(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.cd8: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.3d3(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.7d0: %UInt.as.ImplicitAs.impl.Convert.type.cd8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.256: %ImplicitAs.type.7cb = facet_value %u64, (%ImplicitAs.impl_witness.faa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.802: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.256) [concrete]
// CHECK:STDOUT:   %.2f3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.802, %ImplicitAs.facet.256 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.01f, %UInt.as.ImplicitAs.impl.Convert.7d0 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.7d0, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.b37: <bound method> = bound_method %int_1.01f, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2fa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fcd: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.fcd) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7cb, %ImplicitAs.facet.fcd [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f01 = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT:   %Core.import_ref.baa: @UInt.as.ImplicitAs.impl.3d3.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.3ec) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.3d3.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.06e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.465 = impl_witness_table (%Core.import_ref.baa), @UInt.as.ImplicitAs.impl.3d3 [concrete]
// CHECK:STDOUT:   %Core.import_ref.59a: @Core.IntLiteral.as.As.impl.e51.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.fb3) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.e51.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.f86)]
// CHECK:STDOUT:   %As.impl_witness_table.637 = impl_witness_table (%Core.import_ref.59a), @Core.IntLiteral.as.As.impl.e51 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed5 = impl_witness_access constants.%ImplicitAs.impl_witness.8d9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_46.2(%int_1.loc8) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc8_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc8_46.2: %u64 = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = wrapper_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern cpp_unsigned_long [concrete = constants.%cpp_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = wrapper_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern carbon_unsigned_long [concrete = constants.%carbon_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.7fe = impl_witness_access constants.%ImplicitAs.impl_witness.2fa, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
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
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.b3c = impl_witness_access constants.%As.impl_witness.5477, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.990]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.318]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.2f3 = impl_witness_access constants.%ImplicitAs.impl_witness.faa, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.7d0]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.b37]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %cpp_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.357: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %cpp_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.662: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %cpp_compat_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_compat_long_long [concrete]
// CHECK:STDOUT:   %cpp_compat_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %carbon_i64.patt: %pattern_type.f57 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %carbon_i64_result.patt: %pattern_type.ae7 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   %ptr.aec: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.d39: type = facet_type <@As, @As(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.0f7: <witness> = impl_witness imports.%As.impl_witness_table.27c [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d39 = facet_value Core.IntLiteral, (%As.impl_witness.0f7) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.702: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.d3f: type = fn_type_with_self_type %As.WithSelf.Convert.type.702, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.859 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c2c: %ImplicitAs.type.7cb = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.9c1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.0ca: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.c2c) [concrete]
// CHECK:STDOUT:   %.3f2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.0ca, %ImplicitAs.facet.c2c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.b19, %Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2fa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fcd: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.fcd) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7cb, %ImplicitAs.facet.fcd [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
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
// CHECK:STDOUT:   %As.impl_witness_table.27c = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.084 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.859 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.long_long.as.ImplicitAs.impl.c90 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc15_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc15_38.2: %Cpp.long_long = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = wrapper_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_long_long [concrete = constants.%cpp_long_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16_87.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.662 = addr_of %.loc16_87.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.1(%cpp_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_87.2: init %LongLongResult to %.loc16_87.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_87.3: ref %LongLongResult = temporary %.loc16_87.1, %.loc16_87.2
// CHECK:STDOUT:   %.loc16_87.4: %LongLongResult = acquire_value %.loc16_87.3
// CHECK:STDOUT:   %.loc16_39: type = splice_block %LongLongResult.ref.loc16 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc16: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = wrapper_binding cpp_long_long_result, %.loc16_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_long_long_result [concrete = constants.%cpp_long_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc18: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %LongLong64.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.f0dbf9.1 [concrete = imports.%CppCompat.f0dbf9.1]
// CHECK:STDOUT:     %LongLong64.ref: type = name_ref LongLong64, imports.%Core.LongLong64 [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long: %Cpp.long_long = wrapper_binding cpp_compat_long_long, %cpp_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_compat_long_long [concrete = constants.%cpp_compat_long_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc19_101.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.662 = addr_of %.loc19_101.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.1(%cpp_compat_long_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_101.2: init %LongLongResult to %.loc19_101.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_101.3: ref %LongLongResult = temporary %.loc19_101.1, %.loc19_101.2
// CHECK:STDOUT:   %.loc19_101.4: %LongLongResult = acquire_value %.loc19_101.3
// CHECK:STDOUT:   %.loc19_46: type = splice_block %LongLongResult.ref.loc19 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc19: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = wrapper_binding cpp_compat_long_long_result, %.loc19_101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_compat_long_long_result [concrete = constants.%cpp_compat_long_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc21_42: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc21_39.1: %i64 = as_compatible %cpp_long_long.ref.loc21
// CHECK:STDOUT:   %.loc21_39.2: %i64 = converted %cpp_long_long.ref.loc21, %.loc21_39.1
// CHECK:STDOUT:   %i64.loc21_19: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = wrapper_binding carbon_i64, %.loc21_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.f57 = value_binding_pattern carbon_i64 [concrete = constants.%carbon_i64.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_77.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.aec = addr_of %.loc22_77.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.2(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_77.2: init %LongResult to %.loc22_77.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_77.3: ref %LongResult = temporary %.loc22_77.1, %.loc22_77.2
// CHECK:STDOUT:   %.loc22_77.4: %LongResult = acquire_value %.loc22_77.3
// CHECK:STDOUT:   %.loc22_36: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = wrapper_binding carbon_i64_result, %.loc22_77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.ae7 = value_binding_pattern carbon_i64_result [concrete = constants.%carbon_i64_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.7fe = impl_witness_access constants.%ImplicitAs.impl_witness.2fa, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
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
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.d3f = impl_witness_access constants.%As.impl_witness.0f7, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.3f2 = impl_witness_access constants.%ImplicitAs.impl_witness.9c1, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1ae: <witness> = impl_witness imports.%Copy.impl_witness_table.d10 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.1ae) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.1b6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.252: type = fn_type_with_self_type %Copy.WithSelf.Op.type.1b6, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d10 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_33: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.252 = impl_witness_access constants.%Copy.impl_witness.1ae, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_33, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc9(%.loc9_33)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %EqWith.type.9fc: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f8a: <witness> = impl_witness imports.%EqWith.impl_witness_table.0b8 [concrete]
// CHECK:STDOUT:   %EqWith.facet.a82: %EqWith.type.9fc = facet_value %Cpp.long_long, (%EqWith.impl_witness.f8a) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d0b: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.a82) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.e51: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.a82) [concrete]
// CHECK:STDOUT:   %.594: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d0b, %EqWith.facet.a82 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1ca: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.3ab: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.e51, %EqWith.facet.a82 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.a6a: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.456: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.b11: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.1aa [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.68f: %OrderedWith.type.456 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.b11) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.4ee: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.68f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3de: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.68f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.41f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.68f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.517: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.68f) [concrete]
// CHECK:STDOUT:   %.315: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.41f, %OrderedWith.facet.68f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.7c8: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.c88: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.4ee, %OrderedWith.facet.68f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.606: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.0fd: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.517, %OrderedWith.facet.68f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3de, %OrderedWith.facet.68f [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a87: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc9: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.0b8 = impl_witness_table (%Core.import_ref.a87, %Core.import_ref.adc9), @Cpp.long_long.as.EqWith.impl.eaa [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e8: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757ec: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32c: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.0e8, %Core.import_ref.757ec, %Core.import_ref.c32c, %Core.import_ref.381), @Cpp.long_long.as.OrderedWith.impl.278 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long_long = wrapper_binding a, %.loc8_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Cpp.long_long = wrapper_binding b, %.loc9_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.594 = impl_witness_access constants.%EqWith.impl_witness.f8a, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.3ab = impl_witness_access constants.%EqWith.impl_witness.f8a, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.315 = impl_witness_access constants.%OrderedWith.impl_witness.b11, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.c88 = impl_witness_access constants.%OrderedWith.impl_witness.b11, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.0fd = impl_witness_access constants.%OrderedWith.impl_witness.b11, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.d5a = impl_witness_access constants.%OrderedWith.impl_witness.b11, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d67: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.bf2: %ImplicitAs.type.0ec = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.ec6d64.1: %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.ec6d64.2: %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.aee: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.527 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d9: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.b39) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.38f: <witness> = impl_witness imports.%EqWith.impl_witness_table.48a, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.c62891.1: %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.c62891.2: %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.651: %EqWith.type.d67 = facet_value %Cpp.long_long, (%EqWith.impl_witness.38f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d36: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.651) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.9bb: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.651) [concrete]
// CHECK:STDOUT:   %.195: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d36, %EqWith.facet.651 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.845748.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.c62891.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.845748.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.c62891.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.561: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.c2b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.561, %ImplicitAs.facet.2d9 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7f6: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.9bb, %EqWith.facet.651 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.00e4d4.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.00e4d4.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.af1: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.692f01.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0057df.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.692f01.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0057df.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.975: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.c78: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.fe4, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.a485d6.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.a485d6.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.233: %OrderedWith.type.af1 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.c78) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.997: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.233) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.943: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.233) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.f31: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.233) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ee5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.233) [concrete]
// CHECK:STDOUT:   %.76d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.f31, %OrderedWith.facet.233 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.c5d4fc.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.c5d4fc.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.7d2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.997, %OrderedWith.facet.233 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.8e4ef9.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.a485d6.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.8e4ef9.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.a485d6.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.b5c: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ee5, %OrderedWith.facet.233 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.989912.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.989912.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.f03: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.943, %OrderedWith.facet.233 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3de39c.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3de39c.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.533: <witness> = impl_witness imports.%EqWith.impl_witness_table.48a, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1d70c3.1: %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.065d96.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1d70c3.2: %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.065d96.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.91e: %EqWith.type.aee = facet_value %Cpp.long_long, (%EqWith.impl_witness.533) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4ae: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.91e) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.d74: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.91e) [concrete]
// CHECK:STDOUT:   %.65b: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.4ae, %EqWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.0f23b0.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.1d70c3.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.0f23b0.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.1d70c3.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.e0c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.d74, %EqWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.08840e.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.065d96.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.08840e.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.065d96.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.9a7: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.fe4, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.c909a5.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.c909a5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.c1d: %OrderedWith.type.975 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.9a7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.863: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.c1d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.2c2: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.c1d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.8ed: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.c1d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d59: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.c1d) [concrete]
// CHECK:STDOUT:   %.6b2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.8ed, %OrderedWith.facet.c1d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.84b722.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.84b722.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.bc2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.863, %OrderedWith.facet.c1d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.5f22b4.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.c909a5.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.5f22b4.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.c909a5.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.1fa: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d59, %OrderedWith.facet.c1d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.aae3bc.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.aae3bc.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.409: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.2c2, %OrderedWith.facet.c1d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb0e9f.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb0e9f.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7e1: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.ec6d64.1)]
// CHECK:STDOUT:   %Core.import_ref.ba29: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.48a = impl_witness_table (%Core.import_ref.7e1, %Core.import_ref.ba29), @Cpp.long_long.as.EqWith.impl.600 [concrete]
// CHECK:STDOUT:   %Core.Equal.cce: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.ec6d64.2)]
// CHECK:STDOUT:   %Core.NotEqual.422: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.527 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.020: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.0057df.1)]
// CHECK:STDOUT:   %Core.import_ref.152: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.1)]
// CHECK:STDOUT:   %Core.import_ref.ba3: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.692f01.1)]
// CHECK:STDOUT:   %Core.import_ref.cea: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.fe4 = impl_witness_table (%Core.import_ref.020, %Core.import_ref.152, %Core.import_ref.ba3, %Core.import_ref.cea), @Cpp.long_long.as.OrderedWith.impl.deb [concrete]
// CHECK:STDOUT:   %Core.Less.868: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.0057df.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.211: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.2)]
// CHECK:STDOUT:   %Core.Greater.91c: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.692f01.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.7ba: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long_long = wrapper_binding a, %.loc9_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.195 = impl_witness_access constants.%EqWith.impl_witness.38f, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.c62891.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.845748.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.1 = specific_constant imports.%Core.Equal.cce, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.c62891.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.f5043e.1 = name_ref Equal, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.c62891.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.845748.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.7f6 = impl_witness_access constants.%EqWith.impl_witness.38f, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.00e4d4.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.1 = specific_constant imports.%Core.NotEqual.422, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.db0c7e.1 = name_ref NotEqual, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.d7a6e1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.00e4d4.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.76d = impl_witness_access constants.%OrderedWith.impl_witness.c78, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.c5d4fc.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.1 = specific_constant imports.%Core.Greater.91c, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.eb31d3.1 = name_ref Greater, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ab0365.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.c5d4fc.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.7d2 = impl_witness_access constants.%OrderedWith.impl_witness.c78, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.a485d6.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.8e4ef9.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.1 = specific_constant imports.%Core.Less.868, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.a485d6.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.82967f.1 = name_ref Less, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.a485d6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.8e4ef9.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.b5c = impl_witness_access constants.%OrderedWith.impl_witness.c78, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.989912.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.1 = specific_constant imports.%Core.GreaterOrEquivalent.7ba, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9f5592.1 = name_ref GreaterOrEquivalent, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.8ff390.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.989912.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.f03 = impl_witness_access constants.%OrderedWith.impl_witness.c78, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3de39c.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.1 = specific_constant imports.%Core.LessOrEquivalent.211, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.8801f7.1 = name_ref LessOrEquivalent, %.loc15_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.397e23.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3de39c.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.65b = impl_witness_access constants.%EqWith.impl_witness.533, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1d70c3.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.0f23b0.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.1 = specific_constant imports.%Core.Equal.cce, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1d70c3.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.b8a20f.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1d70c3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.0f23b0.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.e0c = impl_witness_access constants.%EqWith.impl_witness.533, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.065d96.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.08840e.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.1 = specific_constant imports.%Core.NotEqual.422, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.065d96.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3d956b.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.065d96.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.08840e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.6b2 = impl_witness_access constants.%OrderedWith.impl_witness.9a7, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.84b722.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.1 = specific_constant imports.%Core.Greater.91c, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.a8e4b9.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.0fe70e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.84b722.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.bc2 = impl_witness_access constants.%OrderedWith.impl_witness.9a7, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.c909a5.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.5f22b4.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.1 = specific_constant imports.%Core.Less.868, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.c909a5.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.a66e6f.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.c909a5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.5f22b4.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.1fa = impl_witness_access constants.%OrderedWith.impl_witness.9a7, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.aae3bc.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.1 = specific_constant imports.%Core.GreaterOrEquivalent.7ba, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d4199c.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.31eeab.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.aae3bc.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.409 = impl_witness_access constants.%OrderedWith.impl_witness.9a7, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb0e9f.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.1 = specific_constant imports.%Core.LessOrEquivalent.211, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.187e43.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.40c7fd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.82d) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb0e9f.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.9fc: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.bf2: %ImplicitAs.type.0ec = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b16a99.1: %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a1333e.1: %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b16a99.2: %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a1333e.2: %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.527 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d9: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.b39) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.c23: <witness> = impl_witness imports.%EqWith.impl_witness_table.0fe, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.2cebe9.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.84696f.1: %T.as_type.as.EqWith.impl.Equal.type.2cebe9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b335a1.1: %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.2cebe9.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.84696f.2: %T.as_type.as.EqWith.impl.Equal.type.2cebe9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b335a1.2: %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.b15: %EqWith.type.9fc = facet_value %i64, (%EqWith.impl_witness.c23) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.573: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.b15) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.404: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.b15) [concrete]
// CHECK:STDOUT:   %.09f: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.573, %EqWith.facet.b15 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.65a047.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.84696f.2, @T.as_type.as.EqWith.impl.Equal.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.65a047.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.84696f.1, @T.as_type.as.EqWith.impl.Equal.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.561: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.c2b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.561, %ImplicitAs.facet.2d9 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7f5: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.404, %EqWith.facet.b15 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.42b9f1.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.b335a1.2, @T.as_type.as.EqWith.impl.NotEqual.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.42b9f1.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.b335a1.1, @T.as_type.as.EqWith.impl.NotEqual.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.456: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.878ae3.1: %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c37ef5.1: %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.878ae3.2: %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c37ef5.2: %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.658: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.7c0, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c36421.1: %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.2a695b.1: %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c36421.2: %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.2a695b.2: %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.1e7: %OrderedWith.type.456 = facet_value %i64, (%OrderedWith.impl_witness.658) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.07f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.1e7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.669: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.1e7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d40: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.1e7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.aff: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.1e7) [concrete]
// CHECK:STDOUT:   %.fc5: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d40, %OrderedWith.facet.1e7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d9039.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.2a695b.2, @T.as_type.as.OrderedWith.impl.Greater.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d9039.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.2a695b.1, @T.as_type.as.OrderedWith.impl.Greater.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.4c9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.07f, %OrderedWith.facet.1e7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.7d1c6d.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.c36421.2, @T.as_type.as.OrderedWith.impl.Less.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.7d1c6d.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.c36421.1, @T.as_type.as.OrderedWith.impl.Less.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.ff3: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.aff, %OrderedWith.facet.1e7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.c9f471.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.c9f471.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.ef3: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.669, %OrderedWith.facet.1e7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ef18e5.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ef18e5.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.d69: <witness> = impl_witness imports.%EqWith.impl_witness_table.0fe, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.69ee3b.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bf29af.1: %T.as_type.as.EqWith.impl.Equal.type.69ee3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.1: %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.69ee3b.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bf29af.2: %T.as_type.as.EqWith.impl.Equal.type.69ee3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.2: %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.df1: %EqWith.type.9fc = facet_value Core.IntLiteral, (%EqWith.impl_witness.d69) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.7c8: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.df1) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.611: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.df1) [concrete]
// CHECK:STDOUT:   %.9eb: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.7c8, %EqWith.facet.df1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.d28740.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.bf29af.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.bf29af.2, @T.as_type.as.EqWith.impl.Equal.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.64c646.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.d28740.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.bf29af.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.bf29af.1, @T.as_type.as.EqWith.impl.Equal.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.64c646.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.2 [concrete]
// CHECK:STDOUT:   %.e07: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.611, %EqWith.facet.df1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.3b550c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.2, @T.as_type.as.EqWith.impl.NotEqual.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.88f774.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.3b550c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.e6a6e5.1, @T.as_type.as.EqWith.impl.NotEqual.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.88f774.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.01c: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.7c0, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e99f16.1: %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.5bddac.1: %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e99f16.2: %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.5bddac.2: %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.c2d: %OrderedWith.type.456 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.01c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.443: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.f5f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9fe: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ec4: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %.e0e: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9fe, %OrderedWith.facet.c2d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.e43728.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.5bddac.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.5bddac.2, @T.as_type.as.OrderedWith.impl.Greater.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.50b142.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.e43728.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.5bddac.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.5bddac.1, @T.as_type.as.OrderedWith.impl.Greater.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.50b142.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.2 [concrete]
// CHECK:STDOUT:   %.b9e: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.443, %OrderedWith.facet.c2d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.46f927.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.e99f16.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.e99f16.2, @T.as_type.as.OrderedWith.impl.Less.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.af9538.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.46f927.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.e99f16.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.e99f16.1, @T.as_type.as.OrderedWith.impl.Less.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.af9538.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.2 [concrete]
// CHECK:STDOUT:   %.5c6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ec4, %OrderedWith.facet.c2d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.8313f0.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.d93343.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.8313f0.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.d93343.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.2 [concrete]
// CHECK:STDOUT:   %.972: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.f5f, %OrderedWith.facet.c2d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.85eb0f.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.a6a809.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.85eb0f.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(%ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %bound_method.a6a809.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c48: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.a1333e.1)]
// CHECK:STDOUT:   %Core.import_ref.98d: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.b16a99.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.0fe = impl_witness_table (%Core.import_ref.c48, %Core.import_ref.98d), @T.as_type.as.EqWith.impl.6d3 [concrete]
// CHECK:STDOUT:   %Core.Equal.4f5: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.a1333e.2)]
// CHECK:STDOUT:   %Core.NotEqual.0b2: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.b16a99.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.527 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.97f: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.c37ef5.1)]
// CHECK:STDOUT:   %Core.import_ref.4096: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.1)]
// CHECK:STDOUT:   %Core.import_ref.56a: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.878ae3.1)]
// CHECK:STDOUT:   %Core.import_ref.b00d: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.7c0 = impl_witness_table (%Core.import_ref.97f, %Core.import_ref.4096, %Core.import_ref.56a, %Core.import_ref.b00d), @T.as_type.as.OrderedWith.impl.df2 [concrete]
// CHECK:STDOUT:   %Core.Less.2d9: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.c37ef5.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.d90: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.2)]
// CHECK:STDOUT:   %Core.Greater.c24: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.878ae3.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.8e2: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long_long = wrapper_binding a, %.loc9_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.09f = impl_witness_access constants.%EqWith.impl_witness.c23, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.84696f.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.65a047.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %T.as_type.as.EqWith.impl.Equal.type.2cebe9.1 = specific_constant imports.%Core.Equal.4f5, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.84696f.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.2cebe9.1 = name_ref Equal, %.loc10_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.84696f.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.65a047.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.7f5 = impl_witness_access constants.%EqWith.impl_witness.c23, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.b335a1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.42b9f1.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.1 = specific_constant imports.%Core.NotEqual.0b2, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.b335a1.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.6989f5.1 = name_ref NotEqual, %.loc11_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.b335a1.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.42b9f1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.fc5 = impl_witness_access constants.%OrderedWith.impl_witness.658, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.2a695b.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d9039.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.1 = specific_constant imports.%Core.Greater.c24, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.2a695b.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.f879e3.1 = name_ref Greater, %.loc12_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.2a695b.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d9039.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.4c9 = impl_witness_access constants.%OrderedWith.impl_witness.658, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c36421.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.7d1c6d.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.1 = specific_constant imports.%Core.Less.2d9, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c36421.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.b6a41f.1 = name_ref Less, %.loc13_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c36421.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.7d1c6d.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.ff3 = impl_witness_access constants.%OrderedWith.impl_witness.658, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.c9f471.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.1 = specific_constant imports.%Core.GreaterOrEquivalent.8e2, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.532487.1 = name_ref GreaterOrEquivalent, %.loc14_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7e2f13.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.c9f471.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.ef3 = impl_witness_access constants.%OrderedWith.impl_witness.658, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ef18e5.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.1 = specific_constant imports.%Core.LessOrEquivalent.d90, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.17d27e.1 = name_ref LessOrEquivalent, %.loc15_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1da37a.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(constants.%ImplicitAs.facet.2d9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ef18e5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.9eb = impl_witness_access constants.%EqWith.impl_witness.d69, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bf29af.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.d28740.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.64c646.1]
// CHECK:STDOUT:   %.loc17_5.3: %T.as_type.as.EqWith.impl.Equal.type.69ee3b.1 = specific_constant imports.%Core.Equal.4f5, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bf29af.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.69ee3b.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bf29af.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.d28740.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.dbbf20.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.64c646.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.e07 = impl_witness_access constants.%EqWith.impl_witness.d69, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.e6a6e5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.3b550c.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.88f774.1]
// CHECK:STDOUT:   %.loc18_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.1 = specific_constant imports.%Core.NotEqual.0b2, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.e6a6e5.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.ed9945.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.e6a6e5.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.3b550c.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.f946b5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.88f774.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.e0e = impl_witness_access constants.%OrderedWith.impl_witness.01c, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.5bddac.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.e43728.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.50b142.1]
// CHECK:STDOUT:   %.loc19_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.1 = specific_constant imports.%Core.Greater.c24, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.5bddac.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.6d0e34.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.5bddac.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.e43728.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.58afc5.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.50b142.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.b9e = impl_witness_access constants.%OrderedWith.impl_witness.01c, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e99f16.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.46f927.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.af9538.1]
// CHECK:STDOUT:   %.loc20_5.3: %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.1 = specific_constant imports.%Core.Less.2d9, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e99f16.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.0cf30e.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e99f16.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.46f927.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d850a6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.af9538.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.5c6 = impl_witness_access constants.%OrderedWith.impl_witness.01c, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.8313f0.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.d93343.1]
// CHECK:STDOUT:   %.loc21_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.1 = specific_constant imports.%Core.GreaterOrEquivalent.8e2, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.75cdfe.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4c05ce.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.8313f0.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7ede1f.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.d93343.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.972 = impl_witness_access constants.%OrderedWith.impl_witness.01c, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.85eb0f.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.82d]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.a6a809.1]
// CHECK:STDOUT:   %.loc22_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.1 = specific_constant imports.%Core.LessOrEquivalent.d90, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.20fd09.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.60d9eb.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.85eb0f.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(constants.%ImplicitAs.facet.82d) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ebe566.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.a6a809.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.6c4: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.61d: %T.as_type.as.ImplicitAs.impl.Convert.type.6c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.166: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.bf2: %ImplicitAs.type.0ec = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1ae: <witness> = impl_witness imports.%Copy.impl_witness_table.d10 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.1ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fac, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bc8: %ImplicitAs.type.0ec = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.9c2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.f0c: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.24b, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.991: %AddAssignWith.type.166 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.f0c) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.b30: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet.991) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.b30, %AddAssignWith.facet.991 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.7b4e49.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.7b4e49.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.81c: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.be8: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.be9, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.1cd: %SubAssignWith.type.81c = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.be8) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.1e9: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet.1cd) [concrete]
// CHECK:STDOUT:   %.a6c: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.1e9, %SubAssignWith.facet.1cd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.9ec3e7.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.9ec3e7.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.65c: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.013: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.fb7, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.34272f.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.34272f.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.8fc: %MulAssignWith.type.65c = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.013) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.20f: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet.8fc) [concrete]
// CHECK:STDOUT:   %.138: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.20f, %MulAssignWith.facet.8fc [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.d2823b.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.34272f.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.d2823b.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.34272f.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.d82: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.1db: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.228, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.7e1: %DivAssignWith.type.d82 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.1db) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.e03: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet.7e1) [concrete]
// CHECK:STDOUT:   %.96c: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.e03, %DivAssignWith.facet.7e1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.3de7ae.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.3de7ae.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.3e2: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.166: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.f30, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.1f2: %ModAssignWith.type.3e2 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.166) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.b2d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet.1f2) [concrete]
// CHECK:STDOUT:   %.c4ff: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.b2d, %ModAssignWith.facet.1f2 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.5504e8.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.5504e8.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.2f8: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.4ab: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.390, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.aee: %BitAndAssignWith.type.2f8 = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.4ab) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.53f: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet.aee) [concrete]
// CHECK:STDOUT:   %.4d3: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.53f, %BitAndAssignWith.facet.aee [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.0dcb28.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.0dcb28.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.eac: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.928: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.30d, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.75e: %BitOrAssignWith.type.eac = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.928) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.278: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet.75e) [concrete]
// CHECK:STDOUT:   %.fac: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.278, %BitOrAssignWith.facet.75e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.93d1df.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.93d1df.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.434: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.16d: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.414, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.0b0: %BitXorAssignWith.type.434 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.16d) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.3a8: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet.0b0) [concrete]
// CHECK:STDOUT:   %.ed3: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.3a8, %BitXorAssignWith.facet.0b0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.707457.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.707457.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.509: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.9be: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.080, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.e21: %LeftShiftAssignWith.type.509 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.9be) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.c82: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet.e21) [concrete]
// CHECK:STDOUT:   %.ae9: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.c82, %LeftShiftAssignWith.facet.e21 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.795f62.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.795f62.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.a59: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.2a5: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.af8, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.926: %RightShiftAssignWith.type.a59 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.2a5) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.3ee: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet.926) [concrete]
// CHECK:STDOUT:   %.b01: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.3ee, %RightShiftAssignWith.facet.926 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ddd122.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.bc8) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ddd122.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.bc8) [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fac = impl_witness_table (%Core.import_ref.051), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.24b = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d10 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.14b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.be9 = impl_witness_table (%Core.import_ref.14b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3dd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2)]
// CHECK:STDOUT:   %Core.import_ref.1e7: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.fb7 = impl_witness_table (%Core.import_ref.1e7), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1a0: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2)]
// CHECK:STDOUT:   %Core.import_ref.90c: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.228 = impl_witness_table (%Core.import_ref.90c), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4db: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2)]
// CHECK:STDOUT:   %Core.import_ref.620: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.f30 = impl_witness_table (%Core.import_ref.620), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.978: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2)]
// CHECK:STDOUT:   %Core.import_ref.f6c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.390 = impl_witness_table (%Core.import_ref.f6c), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.96f: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.30d = impl_witness_table (%Core.import_ref.4fa), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.152: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2)]
// CHECK:STDOUT:   %Core.import_ref.0a0: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.414 = impl_witness_table (%Core.import_ref.0a0), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.993: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2)]
// CHECK:STDOUT:   %Core.import_ref.463: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.080 = impl_witness_table (%Core.import_ref.463), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3c4: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2)]
// CHECK:STDOUT:   %Core.import_ref.6d5: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.af8 = impl_witness_table (%Core.import_ref.6d5), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.77b: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var_storage %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.474 = impl_witness_access constants.%AddAssignWith.impl_witness.f0c, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.7b4e49.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3c3ed3.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a9ea4f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.7b4e49.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.a6c = impl_witness_access constants.%SubAssignWith.impl_witness.be8, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.9ec3e7.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.1 = specific_constant imports.%Core.Op.3dd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.ecc32f.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2b0a24.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.9ec3e7.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.138 = impl_witness_access constants.%MulAssignWith.impl_witness.013, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.34272f.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.d2823b.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.1 = specific_constant imports.%Core.Op.1a0, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.34272f.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.ce0a27.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.34272f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.d2823b.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.96c = impl_witness_access constants.%DivAssignWith.impl_witness.1db, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.3de7ae.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.1 = specific_constant imports.%Core.Op.4db, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.f8edd0.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.81a1d1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.3de7ae.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.c4ff = impl_witness_access constants.%ModAssignWith.impl_witness.166, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.5504e8.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.1 = specific_constant imports.%Core.Op.978, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.6001cc.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.e8b848.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.5504e8.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.4d3 = impl_witness_access constants.%BitAndAssignWith.impl_witness.4ab, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.0dcb28.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.1 = specific_constant imports.%Core.Op.96f, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.517dbc.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.961e72.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.0dcb28.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.fac = impl_witness_access constants.%BitOrAssignWith.impl_witness.928, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.93d1df.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.1 = specific_constant imports.%Core.Op.152, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.8f8ead.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.e58fe4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.93d1df.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.ed3 = impl_witness_access constants.%BitXorAssignWith.impl_witness.16d, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.707457.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.1 = specific_constant imports.%Core.Op.993, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7cf10a.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.322da2.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.707457.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.ae9 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.9be, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.795f62.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.1 = specific_constant imports.%Core.Op.3c4, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.90e5a3.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c9d902.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.795f62.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.b01 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.2a5, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.9c2) [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.bc8]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ddd122.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.1 = specific_constant imports.%Core.Op.77b, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bd24cc.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5eb4a1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.bc8) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ddd122.2]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.527 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d9: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.b39) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.057: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.24b, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.a5f: %AddAssignWith.type.d51 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.057) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.a94: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.a5f) [concrete]
// CHECK:STDOUT:   %.0aa: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.a94, %AddAssignWith.facet.a5f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.561: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.c2b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.561, %ImplicitAs.facet.2d9 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.01e: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.b1b: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.be9, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.a86429.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.a86429.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.5cd: %SubAssignWith.type.01e = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.b1b) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.495: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet.5cd) [concrete]
// CHECK:STDOUT:   %.21c0: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.495, %SubAssignWith.facet.5cd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2e266b.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.a86429.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2e266b.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.a86429.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.813: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.b10: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.fb7, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.26a: %MulAssignWith.type.813 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.b10) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3e6: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet.26a) [concrete]
// CHECK:STDOUT:   %.d86: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3e6, %MulAssignWith.facet.26a [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.aeb140.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.aeb140.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.4ac: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.ec5: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.228, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.bdb: %DivAssignWith.type.4ac = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.ec5) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.290: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet.bdb) [concrete]
// CHECK:STDOUT:   %.e98: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.290, %DivAssignWith.facet.bdb [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.24aa96.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.24aa96.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.6a8: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.d17: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.f30, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.33527b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.33527b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.5b1: %ModAssignWith.type.6a8 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.d17) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.f67: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet.5b1) [concrete]
// CHECK:STDOUT:   %.e07: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.f67, %ModAssignWith.facet.5b1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.698e15.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.33527b.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.698e15.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.33527b.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.817: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.367: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.390, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.b19: %BitAndAssignWith.type.817 = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.367) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.cb7: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet.b19) [concrete]
// CHECK:STDOUT:   %.f3d: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.cb7, %BitAndAssignWith.facet.b19 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.3c68e0.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.3c68e0.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.9cc: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.573: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.30d, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.acc: %BitOrAssignWith.type.9cc = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.573) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.dfe: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet.acc) [concrete]
// CHECK:STDOUT:   %.60b: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.dfe, %BitOrAssignWith.facet.acc [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.542169.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.542169.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.801: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.2eb: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.414, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.0e7: %BitXorAssignWith.type.801 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.2eb) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.95e: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet.0e7) [concrete]
// CHECK:STDOUT:   %.3ea: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.95e, %BitXorAssignWith.facet.0e7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.602c79.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.602c79.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.d52: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.63f: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.080, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.69f: %LeftShiftAssignWith.type.d52 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.63f) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.6fd: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet.69f) [concrete]
// CHECK:STDOUT:   %.915: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.6fd, %LeftShiftAssignWith.facet.69f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.108543.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.108543.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.7d8: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.47c: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.af8, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.004: %RightShiftAssignWith.type.7d8 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.47c) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.6c4: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet.004) [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.6c4, %RightShiftAssignWith.facet.004 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.9385c3.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.9385c3.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.24b = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.527 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.14b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.be9 = impl_witness_table (%Core.import_ref.14b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3dd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2)]
// CHECK:STDOUT:   %Core.import_ref.1e7: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.fb7 = impl_witness_table (%Core.import_ref.1e7), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1a0: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2)]
// CHECK:STDOUT:   %Core.import_ref.90c: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.228 = impl_witness_table (%Core.import_ref.90c), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4db: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2)]
// CHECK:STDOUT:   %Core.import_ref.620: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.f30 = impl_witness_table (%Core.import_ref.620), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.978: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2)]
// CHECK:STDOUT:   %Core.import_ref.f6c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.390 = impl_witness_table (%Core.import_ref.f6c), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.96f: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.30d = impl_witness_table (%Core.import_ref.4fa), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.152: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2)]
// CHECK:STDOUT:   %Core.import_ref.0a0: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.414 = impl_witness_table (%Core.import_ref.0a0), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.993: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2)]
// CHECK:STDOUT:   %Core.import_ref.463: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.080 = impl_witness_table (%Core.import_ref.463), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3c4: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2)]
// CHECK:STDOUT:   %Core.import_ref.6d5: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.af8 = impl_witness_table (%Core.import_ref.6d5), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.77b: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.0aa = impl_witness_access constants.%AddAssignWith.impl_witness.057, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.21c0 = impl_witness_access constants.%SubAssignWith.impl_witness.b1b, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.a86429.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2e266b.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.1 = specific_constant imports.%Core.Op.3dd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.a86429.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.71efff.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.a86429.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2e266b.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.d86 = impl_witness_access constants.%MulAssignWith.impl_witness.b10, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.aeb140.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.1 = specific_constant imports.%Core.Op.1a0, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.6aa783.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.7f1754.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.aeb140.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.e98 = impl_witness_access constants.%DivAssignWith.impl_witness.ec5, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.24aa96.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.1 = specific_constant imports.%Core.Op.4db, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.31fea9.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.acd96d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.24aa96.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.e07 = impl_witness_access constants.%ModAssignWith.impl_witness.d17, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.33527b.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.698e15.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.1 = specific_constant imports.%Core.Op.978, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.33527b.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.76306f.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.33527b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.698e15.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.f3d = impl_witness_access constants.%BitAndAssignWith.impl_witness.367, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc16_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc16_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc16_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.3c68e0.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.1 = specific_constant imports.%Core.Op.96f, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1531ae.1 = name_ref Op, %.loc16_5.3 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.c2e480.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.3c68e0.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.60b = impl_witness_access constants.%BitOrAssignWith.impl_witness.573, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.542169.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.1 = specific_constant imports.%Core.Op.152, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.7ba83d.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.b8a20c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.542169.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.3ea = impl_witness_access constants.%BitXorAssignWith.impl_witness.2eb, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.602c79.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.1 = specific_constant imports.%Core.Op.993, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1cb851.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.d5755c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.602c79.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.915 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.63f, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.108543.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.1 = specific_constant imports.%Core.Op.3c4, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f6ff82.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.c56168.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.108543.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.b40 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.47c, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.9385c3.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.1 = specific_constant imports.%Core.Op.77b, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.175017.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.110511.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.9385c3.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %AddAssignWith.impl_witness.022: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.24b, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.ff3: %AddAssignWith.type.5ab = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.022) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.498: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.ff3) [concrete]
// CHECK:STDOUT:   %.9a6: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.498, %AddAssignWith.facet.ff3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cb43dd.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cb43dd.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.24b = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.9a6 = impl_witness_access constants.%AddAssignWith.impl_witness.022, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f95) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cb43dd.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.01ebb2.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0ea592.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cb43dd.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82d: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.82d) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet.82d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.f57 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.227: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.235: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.227) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.235) [concrete]
// CHECK:STDOUT:   %.03f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da7, %ImplicitAs.facet.235 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.527 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d9: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.b39) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.057: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.24b, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.a5f: %AddAssignWith.type.d51 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.057) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.a94: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.a5f) [concrete]
// CHECK:STDOUT:   %.0aa: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.a94, %AddAssignWith.facet.a5f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.561: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2d9) [concrete]
// CHECK:STDOUT:   %.c2b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.561, %ImplicitAs.facet.2d9 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.24b = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.527 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.03f = impl_witness_access constants.%ImplicitAs.impl_witness.227, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %b: %i64 = wrapper_binding b, %.loc9_16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f57 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.0aa = impl_witness_access constants.%AddAssignWith.impl_witness.057, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.b39) [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.2d9]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8e5700.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.565cc5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.2d9) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.116bb8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c2b = impl_witness_access constants.%ImplicitAs.impl_witness.b39, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f95: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6d4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f95) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.750: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.750, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.d92: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.471: type = fn_type_with_self_type %Inc.WithSelf.Op.type.d92, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.22e: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %Dec.WithSelf.Op.type.22e, %Dec.facet [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6d4 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long_long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cd7 = impl_witness_access constants.%ImplicitAs.impl_witness.f95, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.471 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.89c = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
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
// CHECK:STDOUT:   %cpp_unsigned_long_long.patt: %pattern_type.4cd = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c8: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d72: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.961 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.745: %ImplicitAs.type.1c8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d72) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.469: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.745) [concrete]
// CHECK:STDOUT:   %.2fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.469, %ImplicitAs.facet.745 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.877: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.130: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %cpp_compat_ulong_long.patt: %pattern_type.4cd = value_binding_pattern cpp_compat_ulong_long [concrete]
// CHECK:STDOUT:   %cpp_compat_ulong_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_compat_ulong_long_result [concrete]
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %carbon_u64.patt: %pattern_type.b35 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8d9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f01, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.633: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e08: %ImplicitAs.type.14f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.8d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.17c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.e08) [concrete]
// CHECK:STDOUT:   %.ed5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.17c, %ImplicitAs.facet.e08 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.633 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.633, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.7e7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.01f: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97f: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %carbon_u64_result.patt: %pattern_type.97f = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   %ptr.e40: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.7f6: type = facet_type <@As, @As(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.0db: <witness> = impl_witness imports.%As.impl_witness_table.701 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.7f6 = facet_value Core.IntLiteral, (%As.impl_witness.0db) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.381: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.94e: type = fn_type_with_self_type %As.WithSelf.Convert.type.381, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8ae: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.faa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a64: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.8ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7b8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.a64) [concrete]
// CHECK:STDOUT:   %.88d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7b8, %ImplicitAs.facet.a64 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.877, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2fa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fcd: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.fcd) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7cb, %ImplicitAs.facet.fcd [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.961 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.f5b [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.aa5: @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.170)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f01 = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.701 = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.f3d [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.faa = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long_long.as.ImplicitAs.impl.437 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.2fd = impl_witness_access constants.%ImplicitAs.impl_witness.d72, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc15_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc15_56.2: %Cpp.unsigned_long_long = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc15: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = wrapper_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.4cd = value_binding_pattern cpp_unsigned_long_long [concrete = constants.%cpp_unsigned_long_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc16: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.130 = addr_of %.loc16_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%cpp_unsigned_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_107.2: init %ULongLongResult to %.loc16_107.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_107.3: ref %ULongLongResult = temporary %.loc16_107.1, %.loc16_107.2
// CHECK:STDOUT:   %.loc16_107.4: %ULongLongResult = acquire_value %.loc16_107.3
// CHECK:STDOUT:   %.loc16_48: type = splice_block %ULongLongResult.ref.loc16 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc16: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = wrapper_binding cpp_unsigned_long_long_result, %.loc16_107.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_unsigned_long_long_result [concrete = constants.%cpp_unsigned_long_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc18: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULongLong64.ref [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.f0dbf9.1 [concrete = imports.%CppCompat.f0dbf9.1]
// CHECK:STDOUT:     %ULongLong64.ref: type = name_ref ULongLong64, imports.%Core.ULongLong64 [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long: %Cpp.unsigned_long_long = wrapper_binding cpp_compat_ulong_long, %cpp_unsigned_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long.patt: %pattern_type.4cd = value_binding_pattern cpp_compat_ulong_long [concrete = constants.%cpp_compat_ulong_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc19_105.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.130 = addr_of %.loc19_105.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%cpp_compat_ulong_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_105.2: init %ULongLongResult to %.loc19_105.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_105.3: ref %ULongLongResult = temporary %.loc19_105.1, %.loc19_105.2
// CHECK:STDOUT:   %.loc19_105.4: %ULongLongResult = acquire_value %.loc19_105.3
// CHECK:STDOUT:   %.loc19_47: type = splice_block %ULongLongResult.ref.loc19 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc19: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = wrapper_binding cpp_compat_ulong_long_result, %.loc19_105.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_compat_ulong_long_result [concrete = constants.%cpp_compat_ulong_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc21: %.ed5 = impl_witness_access constants.%ImplicitAs.impl_witness.8d9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_u64: %u64 = wrapper_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.b35 = value_binding_pattern carbon_u64 [concrete = constants.%carbon_u64.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_79.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.e40 = addr_of %.loc22_79.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.2(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_79.2: init %ULongResult to %.loc22_79.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_79.3: ref %ULongResult = temporary %.loc22_79.1, %.loc22_79.2
// CHECK:STDOUT:   %.loc22_79.4: %ULongResult = acquire_value %.loc22_79.3
// CHECK:STDOUT:   %.loc22_36: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = wrapper_binding carbon_u64_result, %.loc22_79.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.97f = value_binding_pattern carbon_u64_result [concrete = constants.%carbon_u64_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.7fe = impl_witness_access constants.%ImplicitAs.impl_witness.2fa, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
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
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.94e = impl_witness_access constants.%As.impl_witness.0db, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.88d = impl_witness_access constants.%ImplicitAs.impl_witness.8ae, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %a.patt: %pattern_type.4cd = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.4cd = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c8: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d72: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.961 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.1c8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d72) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.469: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.469, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.877: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.4cd = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.4cd = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2e6: <witness> = impl_witness imports.%Copy.impl_witness_table.6d7 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.2e6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.761: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e48: type = fn_type_with_self_type %Copy.WithSelf.Op.type.761, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.961 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.f5b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.6d7 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2fd = impl_witness_access constants.%ImplicitAs.impl_witness.d72, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.877]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.4cd = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.4cd = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long_long = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_42: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.e48 = impl_witness_access constants.%Copy.impl_witness.2e6, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_42, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc9(%.loc9_42)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc9: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.4cd = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.4cd = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
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
