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
// CHECK:STDOUT:     .long = @F.%i64.2
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
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.95b = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.61e = impl_witness_access constants.%ImplicitAs.impl_witness.0a6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.290]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.ec2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_28.2(%int_1.loc8) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_28.2: %i64 = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long: %i64 = value_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.95b = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_50: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn.loc11_50 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_50.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_50.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %f32.97e to %.loc11_50.3 = in_place_init %.loc11_50.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_50.5: init %array_type to %a.var = array_init (%.loc11_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_50.1, %.loc11_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, %i64.2 [concrete = constants.%i64]
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
// CHECK:STDOUT: --- unsigned_long.carbon
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
// CHECK:STDOUT:     .unsigned_long = @F.%u64.2
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
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.d53 = impl_witness_access constants.%ImplicitAs.impl_witness.e55, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.34c]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.4a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_46.2(%int_1.loc8) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_46.2: %u64 = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = value_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_59: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn.loc11_59 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_59.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_59.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_59.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_59.4: init %f32.97e to %.loc11_59.3 = in_place_init %.loc11_59.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_59.5: init %array_type to %a.var = array_init (%.loc11_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_59.1, %.loc11_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
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
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.257: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.a95: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.aa8bde.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.9998b6.1: %PassLongLong__carbon_thunk.type.aa8bde.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.cd9: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %ptr.305: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.aa8bde.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.9998b6.2: %PassLongLong__carbon_thunk.type.aa8bde.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.b3e: type = facet_type <@As, @As(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.815: <witness> = impl_witness imports.%As.impl_witness_table.247 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b3e = facet_value Core.IntLiteral, (%As.impl_witness.815) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.469: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.58e: type = fn_type_with_self_type %As.WithSelf.Convert.type.469, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.68a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d95 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.334: %ImplicitAs.type.139 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.68a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c42: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.334) [concrete]
// CHECK:STDOUT:   %.964: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c42, %ImplicitAs.facet.334 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.092, %Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     .LongLongResult = %LongLongResult.decl
// CHECK:STDOUT:     .PassLongLong = %PassLongLong.cpp_overload_set.value
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .LongLong64 = %Core.LongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.LongLong64: type = import_ref Core//prelude/types/cpp/int, LongLong64, loaded [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.139b1c.1: %PassLongLong__carbon_thunk.type.aa8bde.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.9998b6.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.139b1c.2: %PassLongLong__carbon_thunk.type.aa8bde.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.9998b6.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.38e: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.247 = impl_witness_table (%Core.import_ref.38e), @Core.IntLiteral.as.As.impl.80f [concrete]
// CHECK:STDOUT:   %Core.import_ref.97b: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d95 = impl_witness_table (%Core.import_ref.97b), @Cpp.long_long.as.ImplicitAs.impl.e46 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df4: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df4), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.76e = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc15_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc15_38.2: %Cpp.long_long = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = value_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.257 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16_87.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.a95 = addr_of %.loc16_87.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_87.2: init %LongLongResult to %.loc16_87.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_39: type = splice_block %LongLongResult.ref.loc16 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc16: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_87.3: ref %LongLongResult = temporary %.loc16_87.1, %.loc16_87.2
// CHECK:STDOUT:   %.loc16_87.4: %LongLongResult = acquire_value %.loc16_87.3
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = value_binding cpp_long_long_result, %.loc16_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long.patt: %pattern_type.76e = value_binding_pattern cpp_compat_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc18: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %LongLong64.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %LongLong64.ref: type = name_ref LongLong64, imports.%Core.LongLong64 [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long: %Cpp.long_long = value_binding cpp_compat_long_long, %cpp_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.257 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc19_101.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.a95 = addr_of %.loc19_101.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_compat_long_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_101.2: init %LongLongResult to %.loc19_101.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_46: type = splice_block %LongLongResult.ref.loc19 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc19: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_101.3: ref %LongLongResult = temporary %.loc19_101.1, %.loc19_101.2
// CHECK:STDOUT:   %.loc19_101.4: %LongLongResult = acquire_value %.loc19_101.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = value_binding cpp_compat_long_long_result, %.loc19_101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc21_42: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc21_39.1: %i64 = as_compatible %cpp_long_long.ref.loc21
// CHECK:STDOUT:   %.loc21_39.2: %i64 = converted %cpp_long_long.ref.loc21, %.loc21_39.1
// CHECK:STDOUT:   %i64.loc21_19: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc21_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.cd9 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_77.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.305 = addr_of %.loc22_77.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.2(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_77.2: init %LongResult to %.loc22_77.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_77.3: ref %LongResult = temporary %.loc22_77.1, %.loc22_77.2
// CHECK:STDOUT:   %.loc22_77.4: %LongResult = acquire_value %.loc22_77.3
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = value_binding carbon_i64_result, %.loc22_77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_55.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_55.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_55.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_55.4: init %f32.97e to %.loc24_55.3 = in_place_init %.loc24_55.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_55.5: init %array_type to %a.var = array_init (%.loc24_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_55.1, %.loc24_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.58e = impl_witness_access constants.%As.impl_witness.815, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.964 = impl_witness_access constants.%ImplicitAs.impl_witness.68a, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
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
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.8b0: <witness> = impl_witness imports.%Copy.impl_witness_table.991 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.8b0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.766: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ac1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.766, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.287: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.991 = impl_witness_table (%Core.import_ref.287), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.76e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_33: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.ac1 = impl_witness_access constants.%Copy.impl_witness.8b0, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_33, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc9(%.loc9_33)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.891: <witness> = impl_witness imports.%EqWith.impl_witness_table.dc5 [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.4be = facet_value %Cpp.long_long, (%EqWith.impl_witness.891) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.ca7: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.41f: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.b62: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.ca7, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.5dc: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1f3: %Cpp.long_long.as.EqWith.impl.Equal.type.5dc = struct_value () [concrete]
// CHECK:STDOUT:   %.81b: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.41f, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.282: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.8e9: %Cpp.long_long.as.EqWith.impl.NotEqual.type.282 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f2c: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.547 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.65e = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.f2c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.6ab: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.b52: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.969: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.c71: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %.ba7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.969, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.366: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.98e: %Cpp.long_long.as.OrderedWith.impl.Greater.type.366 = struct_value () [concrete]
// CHECK:STDOUT:   %.3bb: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.6ab, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.14b: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.806: %Cpp.long_long.as.OrderedWith.impl.Less.type.14b = struct_value () [concrete]
// CHECK:STDOUT:   %.ab6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.c71, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.58e: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.90e: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.58e = struct_value () [concrete]
// CHECK:STDOUT:   %.ae4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.b52, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cba: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.01d: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cba = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.a0b: %Cpp.long_long.as.EqWith.impl.Equal.type.5dc = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1f3]
// CHECK:STDOUT:   %Core.import_ref.6a5: %Cpp.long_long.as.EqWith.impl.NotEqual.type.282 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.8e9]
// CHECK:STDOUT:   %EqWith.impl_witness_table.dc5 = impl_witness_table (%Core.import_ref.a0b, %Core.import_ref.6a5), @Cpp.long_long.as.EqWith.impl.4a8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.94e: %Cpp.long_long.as.OrderedWith.impl.Less.type.14b = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.806]
// CHECK:STDOUT:   %Core.import_ref.cf3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.01d]
// CHECK:STDOUT:   %Core.import_ref.5e8: %Cpp.long_long.as.OrderedWith.impl.Greater.type.366 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.98e]
// CHECK:STDOUT:   %Core.import_ref.f97: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.58e = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.90e]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.547 = impl_witness_table (%Core.import_ref.94e, %Core.import_ref.cf3, %Core.import_ref.5e8, %Core.import_ref.f97), @Cpp.long_long.as.OrderedWith.impl.b28 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc8_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.b62 = impl_witness_access constants.%EqWith.impl_witness.891, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1f3]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.81b = impl_witness_access constants.%EqWith.impl_witness.891, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.8e9]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.ba7 = impl_witness_access constants.%OrderedWith.impl_witness.f2c, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.98e]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.3bb = impl_witness_access constants.%OrderedWith.impl_witness.f2c, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.806]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.ab6 = impl_witness_access constants.%OrderedWith.impl_witness.f2c, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.90e]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.ae4 = impl_witness_access constants.%OrderedWith.impl_witness.f2c, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.01d]
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
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d68: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.ea5: %ImplicitAs.type.a03 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.77f(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.926cc9.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.77f(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.926cc9.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.77f(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.355396.1: %Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.77f(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.355396.2: %Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f81: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.798 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cee: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.f81) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f86: <witness> = impl_witness imports.%EqWith.impl_witness_table.b07, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.427dee.1: %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.427dee.2: %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.7a7: %EqWith.type.d68 = facet_value %Cpp.long_long, (%EqWith.impl_witness.f86) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.df2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.7a7) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.8d0: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.7a7) [concrete]
// CHECK:STDOUT:   %.bf0: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.df2, %EqWith.facet.7a7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.ee7dfc.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.427dee.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.479: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.479, %ImplicitAs.facet.cee [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.ee7dfc.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.427dee.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.69d: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.8d0, %EqWith.facet.7a7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.57584c.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.57584c.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.e34: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.45893b.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.45893b.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.215845.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.215845.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.acc636.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.acc636.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b951f5.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.045(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b951f5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.641: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.b88, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.bd7: %OrderedWith.type.e34 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.641) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.7dd: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.bd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.4d1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.bd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.fde: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.bd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.933: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.bd7) [concrete]
// CHECK:STDOUT:   %.f91: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.fde, %OrderedWith.facet.bd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.beeec0.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.beeec0.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.8f9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.7dd, %OrderedWith.facet.bd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.d5e38b.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.d5e38b.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.e14: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.933, %OrderedWith.facet.bd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.373989.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.373989.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.67d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.4d1, %OrderedWith.facet.bd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ada5a.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ada5a.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.5fb: <witness> = impl_witness imports.%EqWith.impl_witness_table.b07, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.1: %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.2: %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.77f(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.3f5: %EqWith.type.ded = facet_value %Cpp.long_long, (%EqWith.impl_witness.5fb) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.952: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.3f5) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.d9e: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.3f5) [concrete]
// CHECK:STDOUT:   %.28b: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.952, %EqWith.facet.3f5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a1b1bd.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a1b1bd.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.29e: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.d9e, %EqWith.facet.3f5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.c998d3.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.c998d3.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.dc3: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.b88, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b1e379.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b1e379.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.045(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.972: %OrderedWith.type.358 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.dc3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.6b3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.972) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.26d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.972) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.b81: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.972) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a45: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.972) [concrete]
// CHECK:STDOUT:   %.eed: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.b81, %OrderedWith.facet.972 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.0c8019.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.0c8019.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.6ec: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.6b3, %OrderedWith.facet.972 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.f9b388.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.b1e379.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.f9b388.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.b1e379.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.2b6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a45, %OrderedWith.facet.972 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4caca4.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4caca4.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.e74: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.26d, %OrderedWith.facet.972 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.23c3e4.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.23c3e4.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.846: @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.355396.1)]
// CHECK:STDOUT:   %Core.import_ref.972: @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.926cc9.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.b07 = impl_witness_table (%Core.import_ref.846, %Core.import_ref.972), @Cpp.long_long.as.EqWith.impl.77f [concrete]
// CHECK:STDOUT:   %Core.NotEqual.e6a: @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.442fe7.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.926cc9.2)]
// CHECK:STDOUT:   %Core.Equal.01c: @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.c14bff.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.77f.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.355396.2)]
// CHECK:STDOUT:   %Core.import_ref.22b: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.798 = impl_witness_table (%Core.import_ref.22b), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.803: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.b951f5.1)]
// CHECK:STDOUT:   %Core.import_ref.37e: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.acc636.1)]
// CHECK:STDOUT:   %Core.import_ref.eb5: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.215845.1)]
// CHECK:STDOUT:   %Core.import_ref.b23: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.45893b.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.b88 = impl_witness_table (%Core.import_ref.803, %Core.import_ref.37e, %Core.import_ref.eb5, %Core.import_ref.b23), @Cpp.long_long.as.OrderedWith.impl.045 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.703: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d1a9aa.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.45893b.2)]
// CHECK:STDOUT:   %Core.Greater.a63: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.77102b.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.215845.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.9d9: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2f4487.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.acc636.2)]
// CHECK:STDOUT:   %Core.Less.d02: @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a8ef98.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.045.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.b951f5.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.bf0 = impl_witness_access constants.%EqWith.impl_witness.f86, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.427dee.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.ee7dfc.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.1 = specific_constant imports.%Core.Equal.01c, @Cpp.long_long.as.EqWith.impl.77f(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.427dee.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.9631ec.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.427dee.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.ee7dfc.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.69d = impl_witness_access constants.%EqWith.impl_witness.f86, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.57584c.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.1 = specific_constant imports.%Core.NotEqual.e6a, @Cpp.long_long.as.EqWith.impl.77f(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.3c7e07.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.80e46f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.57584c.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.f91 = impl_witness_access constants.%OrderedWith.impl_witness.641, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.beeec0.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.1 = specific_constant imports.%Core.Greater.a63, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.fdbe72.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.40b6dd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.beeec0.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.8f9 = impl_witness_access constants.%OrderedWith.impl_witness.641, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.d5e38b.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.1 = specific_constant imports.%Core.Less.d02, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.f945a3.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.89c8d6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.d5e38b.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.e14 = impl_witness_access constants.%OrderedWith.impl_witness.641, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.373989.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.1 = specific_constant imports.%Core.GreaterOrEquivalent.703, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c529aa.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.eeed47.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.373989.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.67d = impl_witness_access constants.%OrderedWith.impl_witness.641, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ada5a.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.1 = specific_constant imports.%Core.LessOrEquivalent.9d9, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.eb047c.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2c8646.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long_long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long_long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ada5a.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.28b = impl_witness_access constants.%EqWith.impl_witness.5fb, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a1b1bd.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.1 = specific_constant imports.%Core.Equal.01c, @Cpp.long_long.as.EqWith.impl.77f(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.32b1f1.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.0ed5ea.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a1b1bd.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.29e = impl_witness_access constants.%EqWith.impl_witness.5fb, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.c998d3.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.1 = specific_constant imports.%Core.NotEqual.e6a, @Cpp.long_long.as.EqWith.impl.77f(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.c1279b.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.6a2818.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.c998d3.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.eed = impl_witness_access constants.%OrderedWith.impl_witness.dc3, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.0c8019.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.1 = specific_constant imports.%Core.Greater.a63, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2b4f26.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a4d75f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.0c8019.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.6ec = impl_witness_access constants.%OrderedWith.impl_witness.dc3, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.b1e379.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.f9b388.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.1 = specific_constant imports.%Core.Less.d02, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.b1e379.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.ab5834.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.b1e379.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.f9b388.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.2b6 = impl_witness_access constants.%OrderedWith.impl_witness.dc3, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4caca4.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.1 = specific_constant imports.%Core.GreaterOrEquivalent.703, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.786521.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.b291a3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long_long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long_long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4caca4.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.e74 = impl_witness_access constants.%OrderedWith.impl_witness.dc3, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.c7b]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.23c3e4.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.1 = specific_constant imports.%Core.LessOrEquivalent.9d9, @Cpp.long_long.as.OrderedWith.impl.045(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.55afed.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.9c0cb5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long_long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long_long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c7b) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.23c3e4.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %Cpp.long_long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.4(%a.ref.loc22, %.loc22_8.2)
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
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.ea5: %ImplicitAs.type.a03 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a81d78.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.cf8(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.3e0733.1: %T.as_type.as.EqWith.impl.NotEqual.type.a81d78.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a81d78.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.cf8(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.3e0733.2: %T.as_type.as.EqWith.impl.NotEqual.type.a81d78.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.cc1a8a.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.cf8(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.38217d.1: %T.as_type.as.EqWith.impl.Equal.type.cc1a8a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.cc1a8a.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.cf8(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.38217d.2: %T.as_type.as.EqWith.impl.Equal.type.cc1a8a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f81: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.798 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cee: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.f81) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.5fe: <witness> = impl_witness imports.%EqWith.impl_witness_table.f35, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.19c770.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.7b1855.1: %T.as_type.as.EqWith.impl.Equal.type.19c770.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.98498c.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a3d76f.1: %T.as_type.as.EqWith.impl.NotEqual.type.98498c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.19c770.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.7b1855.2: %T.as_type.as.EqWith.impl.Equal.type.19c770.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.98498c.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a3d76f.2: %T.as_type.as.EqWith.impl.NotEqual.type.98498c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.113: %EqWith.type.4be = facet_value %i64, (%EqWith.impl_witness.5fe) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d57: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.113) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.5cd: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.113) [concrete]
// CHECK:STDOUT:   %.c82: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d57, %EqWith.facet.113 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.da1fdf.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.7b1855.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.da1fdf.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.7b1855.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.479: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.479, %ImplicitAs.facet.cee [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.185: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.5cd, %EqWith.facet.113 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5afea8.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a3d76f.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5afea8.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a3d76f.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f7bcd.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f7bcd.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.9b1682.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6caf33.1: %T.as_type.as.OrderedWith.impl.Greater.type.9b1682.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.9b1682.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6caf33.2: %T.as_type.as.OrderedWith.impl.Greater.type.9b1682.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6c5131.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6c5131.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.921593.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.1e7362.1: %T.as_type.as.OrderedWith.impl.Less.type.921593.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.921593.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.319(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.1e7362.2: %T.as_type.as.OrderedWith.impl.Less.type.921593.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f1e: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.7ae, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.921e4c.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.df14d5.1: %T.as_type.as.OrderedWith.impl.Less.type.921e4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.dd484f.1: %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.921e4c.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.df14d5.2: %T.as_type.as.OrderedWith.impl.Less.type.921e4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.dd484f.2: %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.250: %OrderedWith.type.65e = facet_value %i64, (%OrderedWith.impl_witness.f1e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.995: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.250) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.d85: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.250) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.1c4: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.250) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.68b: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.250) [concrete]
// CHECK:STDOUT:   %.78a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.1c4, %OrderedWith.facet.250 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.488b23.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.dd484f.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.488b23.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.dd484f.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.76a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.995, %OrderedWith.facet.250 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.ef5f51.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.df14d5.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.ef5f51.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.df14d5.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.68b, %OrderedWith.facet.250 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5a4cd1.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5a4cd1.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.e17: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.d85, %OrderedWith.facet.250 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9118db.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9118db.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.c02: <witness> = impl_witness imports.%EqWith.impl_witness_table.f35, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.235a4b.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.16ab40.1: %T.as_type.as.EqWith.impl.Equal.type.235a4b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.fed4da.1: %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.235a4b.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.16ab40.2: %T.as_type.as.EqWith.impl.Equal.type.235a4b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.cf8(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.fed4da.2: %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.65c: %EqWith.type.4be = facet_value Core.IntLiteral, (%EqWith.impl_witness.c02) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.70c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.65c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.3a9: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.65c) [concrete]
// CHECK:STDOUT:   %.14d: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.70c, %EqWith.facet.65c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ba6709.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.16ab40.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.16ab40.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.0e7187.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ba6709.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.16ab40.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.16ab40.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.0e7187.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.2 [concrete]
// CHECK:STDOUT:   %.466: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.3a9, %EqWith.facet.65c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.771cd0.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.fed4da.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.fed4da.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.c4535c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.771cd0.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.fed4da.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.fed4da.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.c4535c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2bc: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.7ae, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.717a23.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.2f32ca.1: %T.as_type.as.OrderedWith.impl.Less.type.717a23.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.7de439.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.426d92.1: %T.as_type.as.OrderedWith.impl.Greater.type.7de439.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.717a23.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.2f32ca.2: %T.as_type.as.OrderedWith.impl.Less.type.717a23.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.7de439.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.426d92.2: %T.as_type.as.OrderedWith.impl.Greater.type.7de439.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.319(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.73a: %OrderedWith.type.65e = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.2bc) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.934: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.73a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.97f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.73a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.f72: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.73a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.0a0: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.73a) [concrete]
// CHECK:STDOUT:   %.bdf: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.f72, %OrderedWith.facet.73a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.35851a.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.426d92.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.426d92.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.b9aed5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.35851a.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.426d92.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.426d92.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.b9aed5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.2 [concrete]
// CHECK:STDOUT:   %.305: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.934, %OrderedWith.facet.73a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.a2bb6b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.2f32ca.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.2f32ca.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.ed8905.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.a2bb6b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.2f32ca.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.2f32ca.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.ed8905.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.2 [concrete]
// CHECK:STDOUT:   %.3bd: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.0a0, %OrderedWith.facet.73a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.00ca46.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.eda767.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.00ca46.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.eda767.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.2 [concrete]
// CHECK:STDOUT:   %.470: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.97f, %OrderedWith.facet.73a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.9debba.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.9e67cf.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.9debba.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %bound_method.9e67cf.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d3: @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.cc1a8a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.38217d.1)]
// CHECK:STDOUT:   %Core.import_ref.24e: @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.a81d78.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.3e0733.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.f35 = impl_witness_table (%Core.import_ref.1d3, %Core.import_ref.24e), @T.as_type.as.EqWith.impl.cf8 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.86b: @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.a81d78.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.3e0733.2)]
// CHECK:STDOUT:   %Core.Equal.caf: @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.cc1a8a.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.cf8.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.38217d.2)]
// CHECK:STDOUT:   %Core.import_ref.22b: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.798 = impl_witness_table (%Core.import_ref.22b), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.972: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.921593.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.1e7362.1)]
// CHECK:STDOUT:   %Core.import_ref.f5b: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6c5131.1)]
// CHECK:STDOUT:   %Core.import_ref.d55: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.9b1682.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.6caf33.1)]
// CHECK:STDOUT:   %Core.import_ref.a4c: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f7bcd.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.7ae = impl_witness_table (%Core.import_ref.972, %Core.import_ref.f5b, %Core.import_ref.d55, %Core.import_ref.a4c), @T.as_type.as.OrderedWith.impl.319 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.21c: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d1b2d9.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f7bcd.2)]
// CHECK:STDOUT:   %Core.Greater.20c: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.9b1682.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.6caf33.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.223: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.824dd7.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6c5131.2)]
// CHECK:STDOUT:   %Core.Less.7b4: @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.921593.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.319.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.1e7362.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.c82 = impl_witness_access constants.%EqWith.impl_witness.5fe, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.7b1855.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.da1fdf.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.19c770.1 = specific_constant imports.%Core.Equal.caf, @T.as_type.as.EqWith.impl.cf8(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.7b1855.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.19c770.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.7b1855.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.da1fdf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.185 = impl_witness_access constants.%EqWith.impl_witness.5fe, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a3d76f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.5afea8.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.98498c.1 = specific_constant imports.%Core.NotEqual.86b, @T.as_type.as.EqWith.impl.cf8(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a3d76f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.98498c.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a3d76f.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.5afea8.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.78a = impl_witness_access constants.%OrderedWith.impl_witness.f1e, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dd484f.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.488b23.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.1 = specific_constant imports.%Core.Greater.20c, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dd484f.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.3e0887.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dd484f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.488b23.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.76a = impl_witness_access constants.%OrderedWith.impl_witness.f1e, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.df14d5.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.ef5f51.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.921e4c.1 = specific_constant imports.%Core.Less.7b4, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.df14d5.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.921e4c.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.df14d5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.ef5f51.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.63b = impl_witness_access constants.%OrderedWith.impl_witness.f1e, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5a4cd1.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.1 = specific_constant imports.%Core.GreaterOrEquivalent.21c, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.855ee9.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.07df61.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5a4cd1.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.e17 = impl_witness_access constants.%OrderedWith.impl_witness.f1e, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9118db.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.1 = specific_constant imports.%Core.LessOrEquivalent.223, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d8b16e.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.06af75.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.cee) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9118db.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.14d = impl_witness_access constants.%EqWith.impl_witness.c02, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.16ab40.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ba6709.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.0e7187.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.235a4b.1 = specific_constant imports.%Core.Equal.caf, @T.as_type.as.EqWith.impl.cf8(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.16ab40.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.235a4b.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.16ab40.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ba6709.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.c2a0d9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.0e7187.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.466 = impl_witness_access constants.%EqWith.impl_witness.c02, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.fed4da.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.771cd0.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.c4535c.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.1 = specific_constant imports.%Core.NotEqual.86b, @T.as_type.as.EqWith.impl.cf8(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.fed4da.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.085ab8.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.fed4da.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.771cd0.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.db442c.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.c4535c.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.bdf = impl_witness_access constants.%OrderedWith.impl_witness.2bc, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.426d92.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.35851a.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.b9aed5.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.7de439.1 = specific_constant imports.%Core.Greater.20c, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.426d92.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.7de439.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.426d92.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.35851a.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.ae2196.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.b9aed5.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.305 = impl_witness_access constants.%OrderedWith.impl_witness.2bc, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2f32ca.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.a2bb6b.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.ed8905.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.717a23.1 = specific_constant imports.%Core.Less.7b4, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2f32ca.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.717a23.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2f32ca.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.a2bb6b.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.819b2c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.ed8905.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.3bd = impl_witness_access constants.%OrderedWith.impl_witness.2bc, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.00ca46.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.eda767.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.1 = specific_constant imports.%Core.GreaterOrEquivalent.21c, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a9fbe.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.425cdf.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.00ca46.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.617ec2.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.eda767.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.470 = impl_witness_access constants.%OrderedWith.impl_witness.2bc, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.9debba.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.9e67cf.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.1 = specific_constant imports.%Core.LessOrEquivalent.223, @T.as_type.as.OrderedWith.impl.319(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.df40e8.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.5fd398.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.9debba.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.c7b) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b7717f.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.9e67cf.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long_long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.092]
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
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.334: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.b8b: %T.as_type.as.ImplicitAs.impl.Convert.type.334 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.2ec: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8b0: <witness> = impl_witness imports.%Copy.impl_witness_table.991 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.8b0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.68d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6e, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.73b: %ImplicitAs.type.a03 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.68d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.bf6: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d67, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.2ec = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.bf6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.03b: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e56: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.03b, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cf84ae.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cf84ae.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b35: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.ca9: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.20b, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.055560.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.055560.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b35 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.ca9) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.010: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.d9d: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.010, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.3f0e56.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.055560.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.3f0e56.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.055560.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.112: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.6f5: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.20a, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.112 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.6f5) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.16b: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3df: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.16b, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.54521e.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.54521e.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.45d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.18b: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.b4b, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.45d = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.18b) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.2f0: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e87: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.2f0, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.615d14.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.615d14.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.a2e: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.a7a: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.087, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.a2e = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.a7a) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.e9d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.1d8: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.e9d, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49870b.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49870b.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.9bc: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.a42: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.6c5, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.9bc = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.a42) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.8bc: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.564: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.8bc, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.1d5894.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.1d5894.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.219: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.7cf: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.f98, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.219 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.2ae: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.620: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.2ae, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.76b01d.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.76b01d.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.f9d: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.0ca: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.1e7, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.f9d = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.0ca) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.af7: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.efd: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.af7, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.960526.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.960526.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.f24: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.bee: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.f0d, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.f24 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.bee) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.5d0: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ecf: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.5d0, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.330db6.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.330db6.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.42e: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.b65: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.7d4, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.42e = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.b65) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.b72: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.580: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.b72, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.e2803d.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.e2803d.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.73b) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.758: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.334) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.b8b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6e = impl_witness_table (%Core.import_ref.758), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d67 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.879: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2)]
// CHECK:STDOUT:   %Core.import_ref.287: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.991 = impl_witness_table (%Core.import_ref.287), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.da2: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.20b = impl_witness_table (%Core.import_ref.da2), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4cd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.2)]
// CHECK:STDOUT:   %Core.import_ref.923: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.20a = impl_witness_table (%Core.import_ref.923), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a97: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.2)]
// CHECK:STDOUT:   %Core.import_ref.dfe: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.b4b = impl_witness_table (%Core.import_ref.dfe), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ce4: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.2)]
// CHECK:STDOUT:   %Core.import_ref.97b: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.087 = impl_witness_table (%Core.import_ref.97b), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f08: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.2)]
// CHECK:STDOUT:   %Core.import_ref.df1: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.6c5 = impl_witness_table (%Core.import_ref.df1), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4fd: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.2)]
// CHECK:STDOUT:   %Core.import_ref.268: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.f98 = impl_witness_table (%Core.import_ref.268), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d7f: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.2)]
// CHECK:STDOUT:   %Core.import_ref.b5d: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.1e7 = impl_witness_table (%Core.import_ref.b5d), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.66f: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.2)]
// CHECK:STDOUT:   %Core.import_ref.076: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.f0d = impl_witness_table (%Core.import_ref.076), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.540: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.2)]
// CHECK:STDOUT:   %Core.import_ref.011: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.7d4 = impl_witness_table (%Core.import_ref.011), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6ce: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.76e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.e56 = impl_witness_access constants.%AddAssignWith.impl_witness.bf6, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cf84ae.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.1 = specific_constant imports.%Core.Op.879, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.764577.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9f5215.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.cf84ae.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.d9d = impl_witness_access constants.%SubAssignWith.impl_witness.ca9, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.055560.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.3f0e56.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.1 = specific_constant imports.%Core.Op.4cd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.055560.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.bbcd3f.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.055560.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.3f0e56.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.3df = impl_witness_access constants.%MulAssignWith.impl_witness.6f5, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.54521e.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.1 = specific_constant imports.%Core.Op.a97, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.930c4a.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.615bbe.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.54521e.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.e87 = impl_witness_access constants.%DivAssignWith.impl_witness.18b, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.615d14.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.1 = specific_constant imports.%Core.Op.ce4, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.0750e7.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.edd3fd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.615d14.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.1d8 = impl_witness_access constants.%ModAssignWith.impl_witness.a7a, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49870b.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.1 = specific_constant imports.%Core.Op.f08, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.423758.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b6ce41.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49870b.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.564 = impl_witness_access constants.%BitAndAssignWith.impl_witness.a42, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.1d5894.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.1 = specific_constant imports.%Core.Op.4fd, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.34c5a7.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.492145.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.1d5894.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.620 = impl_witness_access constants.%BitOrAssignWith.impl_witness.7cf, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.76b01d.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.1 = specific_constant imports.%Core.Op.d7f, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1655c9.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.97ef88.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.76b01d.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.efd = impl_witness_access constants.%BitXorAssignWith.impl_witness.0ca, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.960526.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.1 = specific_constant imports.%Core.Op.66f, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.42f5ff.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.60b652.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.960526.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.ecf = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.bee, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.330db6.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.1 = specific_constant imports.%Core.Op.540, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.c0189e.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e0e938.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.330db6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.580 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.b65, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.68d) [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.73b]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.e2803d.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.1 = specific_constant imports.%Core.Op.6ce, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.440250.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.60f372.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.73b) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.e2803d.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long_long) {
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
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d84: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f81: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.798 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cee: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.f81) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c98: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d67, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.c98) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.9f5: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.605: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.9f5, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.479: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.479, %ImplicitAs.facet.cee [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.d66: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.230: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.20b, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.d66 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.230) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.f81: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.d45: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.f81, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.a1a11e.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.a1a11e.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.e40: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.fd5: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.20a, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.e40 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.fd5) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.51d: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.a7c: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.51d, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.da4954.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.da4954.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.511: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.b7a: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.b4b, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.511 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.b7a) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.a7c: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9e9: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.a7c, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.b4ca85.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.b4ca85.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2c3: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.746: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.087, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2c3 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.746) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.2ef: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.35e: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.2ef, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49ce9c.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49ce9c.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.8af: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.a30: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.6c5, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.8af = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.a30) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.4de: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.6cf: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.4de, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.2fa8b1.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.2fa8b1.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.172: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.f30: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.f98, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.172 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.f30) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.3d7: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.49d: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.3d7, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.bee85f.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.bee85f.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.2c8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.637: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.1e7, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.2c8 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.637) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.b41: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.cd3: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.b41, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.659754.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.659754.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.b97: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.cea: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.f0d, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.b97 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.cea) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.726: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.90e: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.726, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ab2c2b.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ab2c2b.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.721: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.cea: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.7d4, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.721 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.cea) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.1c1: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ec8: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.1c1, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.3074f6.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.3074f6.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d67 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.879: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2)]
// CHECK:STDOUT:   %Core.import_ref.22b: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.798 = impl_witness_table (%Core.import_ref.22b), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.da2: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.20b = impl_witness_table (%Core.import_ref.da2), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4cd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.f70730.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.c4efbe.2)]
// CHECK:STDOUT:   %Core.import_ref.923: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.20a = impl_witness_table (%Core.import_ref.923), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a97: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.2f4f2d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.a551e1.2)]
// CHECK:STDOUT:   %Core.import_ref.dfe: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.b4b = impl_witness_table (%Core.import_ref.dfe), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ce4: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.9006dc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.9a3809.2)]
// CHECK:STDOUT:   %Core.import_ref.97b: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.087 = impl_witness_table (%Core.import_ref.97b), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f08: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.599fab.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.f4ed2a.2)]
// CHECK:STDOUT:   %Core.import_ref.df1: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.6c5 = impl_witness_table (%Core.import_ref.df1), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4fd: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.484b22.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4334b9.2)]
// CHECK:STDOUT:   %Core.import_ref.268: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.f98 = impl_witness_table (%Core.import_ref.268), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d7f: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.b4fec3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.154e3f.2)]
// CHECK:STDOUT:   %Core.import_ref.b5d: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.1e7 = impl_witness_table (%Core.import_ref.b5d), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.66f: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.23e447.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.233cb5.2)]
// CHECK:STDOUT:   %Core.import_ref.076: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.f0d = impl_witness_table (%Core.import_ref.076), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.540: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.9c9b4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.102435.2)]
// CHECK:STDOUT:   %Core.import_ref.011: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.7d4 = impl_witness_table (%Core.import_ref.011), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6ce: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f09acb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5f8319.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.605 = impl_witness_access constants.%AddAssignWith.impl_witness.c98, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = specific_constant imports.%Core.Op.879, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.d45 = impl_witness_access constants.%SubAssignWith.impl_witness.230, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.a1a11e.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.1 = specific_constant imports.%Core.Op.4cd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.e891c3.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.65ca20.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.a1a11e.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.a7c = impl_witness_access constants.%MulAssignWith.impl_witness.fd5, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.da4954.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.1 = specific_constant imports.%Core.Op.a97, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.31c331.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.407d6d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.da4954.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.9e9 = impl_witness_access constants.%DivAssignWith.impl_witness.b7a, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.b4ca85.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.1 = specific_constant imports.%Core.Op.ce4, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.35f78a.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.5f629f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.b4ca85.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.35e = impl_witness_access constants.%ModAssignWith.impl_witness.746, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49ce9c.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.1 = specific_constant imports.%Core.Op.f08, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.710c53.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6cec08.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.49ce9c.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.6cf = impl_witness_access constants.%BitAndAssignWith.impl_witness.a30, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.2fa8b1.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.1 = specific_constant imports.%Core.Op.4fd, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.ef8d37.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.98a24e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long_long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long_long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.2fa8b1.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.49d = impl_witness_access constants.%BitOrAssignWith.impl_witness.f30, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.bee85f.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.1 = specific_constant imports.%Core.Op.d7f, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a11270.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.efcdd8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.bee85f.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.cd3 = impl_witness_access constants.%BitXorAssignWith.impl_witness.637, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.659754.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.1 = specific_constant imports.%Core.Op.66f, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.533a57.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.427b1c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.659754.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.90e = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.cea, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ab2c2b.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.1 = specific_constant imports.%Core.Op.540, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.b8e2a1.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.3761f5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ab2c2b.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.ec8 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.cea, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.3074f6.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.1 = specific_constant imports.%Core.Op.6ce, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.71dc97.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.be7a73.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.3074f6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20_9: init %Cpp.long_long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20_9
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long_long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long_long) {
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
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.d24: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d67, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.65d685.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.65d685.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.d24) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.4c6: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.2d5: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.4c6, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.616d71.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.65d685.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.616d71.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.65d685.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d67 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.879: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.2d5 = impl_witness_access constants.%AddAssignWith.impl_witness.d24, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.65d685.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a3d) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.616d71.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.1 = specific_constant imports.%Core.Op.879, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.65d685.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.9da307.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.65d685.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long_long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long_long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.616d71.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8: init %Cpp.long_long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.4(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long_long) {
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
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c7b: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c7b) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet.c7b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.257(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl.257(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.257(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.290: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5db: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.0a6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.261: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.5db) [concrete]
// CHECK:STDOUT:   %.61e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.261, %ImplicitAs.facet.5db [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.140: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.290 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.290, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d84: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f81: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.798 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cee: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.f81) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c98: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d67, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.c98) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.9f5: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.605: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.9f5, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.479: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.479, %ImplicitAs.facet.cee [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.799 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.257.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.257.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl.257 [concrete]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d67 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.879: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.8cf6ad.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.9ff12b.2)]
// CHECK:STDOUT:   %Core.import_ref.22b: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.798 = impl_witness_table (%Core.import_ref.22b), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.491]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc9: %.61e = impl_witness_access constants.%ImplicitAs.impl_witness.0a6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.290]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.140]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.605 = impl_witness_access constants.%AddAssignWith.impl_witness.c98, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.f81) [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.cee]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = specific_constant imports.%Core.Op.879, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.61f9d4.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.3aaab8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.cee) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.47f55f.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.48b = impl_witness_access constants.%ImplicitAs.impl_witness.f81, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long_long) {
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
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cd8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cd8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.113: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.e05: type = fn_type_with_self_type %Inc.WithSelf.Op.type.113, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.0d7: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.0ae: type = fn_type_with_self_type %Dec.WithSelf.Op.type.0d7, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.type: type = fn_type @Cpp.long_long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op: %Cpp.long_long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d96: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79f = impl_witness_table (%Core.import_ref.d96), @Core.IntLiteral.as.ImplicitAs.impl.67d [concrete]
// CHECK:STDOUT:   %Core.import_ref.89b: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.89b), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bbc: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.bbc), @Cpp.long_long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b89 = impl_witness_access constants.%ImplicitAs.impl_witness.a3d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.e05 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.0ae = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long_long) {
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
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.407: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.584: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c82 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecd: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1d8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.21c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.ecd) [concrete]
// CHECK:STDOUT:   %.021: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.21c, %ImplicitAs.facet.ecd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.816: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.356: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.816 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c3: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.356 [concrete]
// CHECK:STDOUT:   %int_1.79a: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.633: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d19: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.88c6c2.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.887181.1: %PassULongLong__carbon_thunk.type.88c6c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.703(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e55: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a3, @Core.IntLiteral.as.ImplicitAs.impl.703(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f2f: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.703(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f2f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.32a: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e55) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d41: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.32a) [concrete]
// CHECK:STDOUT:   %.d53: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d41, %ImplicitAs.facet.32a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.6f2: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %ptr.f5e: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.88c6c2.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.887181.2: %PassULongLong__carbon_thunk.type.88c6c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.45a: type = facet_type <@As, @As(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.a80: <witness> = impl_witness imports.%As.impl_witness_table.eec [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.45a = facet_value Core.IntLiteral, (%As.impl_witness.a80) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.699: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.905: type = fn_type_with_self_type %As.WithSelf.Convert.type.699, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6e9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8af [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cb8: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.6e9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e1c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.cb8) [concrete]
// CHECK:STDOUT:   %.038: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e1c, %ImplicitAs.facet.cb8 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.79a, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     .ULongLongResult = %ULongLongResult.decl
// CHECK:STDOUT:     .PassULongLong = %PassULongLong.cpp_overload_set.value
// CHECK:STDOUT:     .ULongResult = %ULongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULongLong64 = %Core.ULongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULongLong64: type = import_ref Core//prelude/types/cpp/int, ULongLong64, loaded [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.aa5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.816 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.356]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c82 = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.348 [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.f0c073.1: %PassULongLong__carbon_thunk.type.88c6c2.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.887181.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.2b3: @Core.IntLiteral.as.ImplicitAs.impl.703.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.703.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a3 = impl_witness_table (%Core.import_ref.2b3), @Core.IntLiteral.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.f0c073.2: %PassULongLong__carbon_thunk.type.88c6c2.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.887181.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.0cd: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.eec = impl_witness_table (%Core.import_ref.0cd), @Core.IntLiteral.as.As.impl.d79 [concrete]
// CHECK:STDOUT:   %Core.import_ref.518: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8af = impl_witness_table (%Core.import_ref.518), @Cpp.unsigned_long_long.as.ImplicitAs.impl.5f4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df4: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df4), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.ebd = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc15: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.021 = impl_witness_access constants.%ImplicitAs.impl_witness.1d8, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.356]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc15_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc15_56.2: %Cpp.unsigned_long_long = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = value_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.633 = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc16: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.d19 = addr_of %.loc16_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_unsigned_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_107.2: init %ULongLongResult to %.loc16_107.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_48: type = splice_block %ULongLongResult.ref.loc16 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc16: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_107.3: ref %ULongLongResult = temporary %.loc16_107.1, %.loc16_107.2
// CHECK:STDOUT:   %.loc16_107.4: %ULongLongResult = acquire_value %.loc16_107.3
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = value_binding cpp_unsigned_long_long_result, %.loc16_107.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long.patt: %pattern_type.ebd = value_binding_pattern cpp_compat_ulong_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc18: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULongLong64.ref [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %ULongLong64.ref: type = name_ref ULongLong64, imports.%Core.ULongLong64 [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long: %Cpp.unsigned_long_long = value_binding cpp_compat_ulong_long, %cpp_unsigned_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long_result.patt: %pattern_type.633 = value_binding_pattern cpp_compat_ulong_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc19_105.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.d19 = addr_of %.loc19_105.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_compat_ulong_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_105.2: init %ULongLongResult to %.loc19_105.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_47: type = splice_block %ULongLongResult.ref.loc19 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc19: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_105.3: ref %ULongLongResult = temporary %.loc19_105.1, %.loc19_105.2
// CHECK:STDOUT:   %.loc19_105.4: %ULongLongResult = acquire_value %.loc19_105.3
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = value_binding cpp_compat_ulong_long_result, %.loc19_105.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.157 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc21: %.d53 = impl_witness_access constants.%ImplicitAs.impl_witness.e55, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.34c]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62d]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.4a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %carbon_u64: %u64 = value_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.6f2 = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_79.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.f5e = addr_of %.loc22_79.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.2(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_79.2: init %ULongResult to %.loc22_79.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_79.3: ref %ULongResult = temporary %.loc22_79.1, %.loc22_79.2
// CHECK:STDOUT:   %.loc22_79.4: %ULongResult = acquire_value %.loc22_79.3
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = value_binding carbon_u64_result, %.loc22_79.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc24_64.1: <bound method> = bound_method %float, %impl.elem0.loc24_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_64.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_64.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_64.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_64.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_64.4: init %f32.97e to %.loc24_64.3 = in_place_init %.loc24_64.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_64.5: init %array_type to %a.var = array_init (%.loc24_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_64.1, %.loc24_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc24: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.905 = impl_witness_access constants.%As.impl_witness.a80, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.038 = impl_witness_access constants.%ImplicitAs.impl_witness.6e9, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
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
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.407: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c82 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1d8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.21c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.021: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.21c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.79a: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d4e: <witness> = impl_witness imports.%Copy.impl_witness_table.dac [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.d4e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.211: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2c9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.211, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.aa5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c82 = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.348 [concrete]
// CHECK:STDOUT:   %Core.import_ref.46e: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.dac = impl_witness_table (%Core.import_ref.46e), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ebd = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ebd = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.021 = impl_witness_access constants.%ImplicitAs.impl_witness.1d8, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ebd = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ebd = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_42: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.2c9 = impl_witness_access constants.%Copy.impl_witness.d4e, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_42, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc9(%.loc9_42)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc9: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %u64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.unsigned_long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
