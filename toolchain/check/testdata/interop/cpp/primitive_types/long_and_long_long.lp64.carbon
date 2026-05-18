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
// CHECK:STDOUT:     .long = @F.%i64.2
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
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.c8a = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.036 = impl_witness_access constants.%ImplicitAs.impl_witness.a33, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.3b9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_28.2(%int_1.loc8) [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc8_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc8_28.2: %i64 = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %cpp_long: %i64 = value_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.c8a = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_50: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn.loc11_50 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc11_50.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_50.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %f32.750 to %.loc11_50.3 = in_place_init %.loc11_50.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_50.5: init %array_type to %a.var = array_init (%.loc11_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_50.1, %.loc11_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, %i64.2 [concrete = constants.%i64]
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
// CHECK:STDOUT: --- unsigned_long.carbon
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
// CHECK:STDOUT:     .unsigned_long = @F.%u64.2
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
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.bb8 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f05 = impl_witness_access constants.%ImplicitAs.impl_witness.093, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.818]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_46.2(%int_1.loc8) [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc8_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc8_46.2: %u64 = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = value_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.bb8 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_59: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn.loc11_59 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc11_59.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_59.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_59.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_59.4: init %f32.750 to %.loc11_59.3 = in_place_init %.loc11_59.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_59.5: init %array_type to %a.var = array_init (%.loc11_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_59.1, %.loc11_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.a44: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.a89: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.322742.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.ad287c.1: %PassLongLong__carbon_thunk.type.322742.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.5f3: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %ptr.618: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.322742.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.ad287c.2: %PassLongLong__carbon_thunk.type.322742.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.fa8: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.cd2: type = facet_type <@As, @As(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.bf0: <witness> = impl_witness imports.%As.impl_witness_table.d2b [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.cd2 = facet_value Core.IntLiteral, (%As.impl_witness.bf0) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a0a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.c3e: type = fn_type_with_self_type %As.WithSelf.Convert.type.a0a, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4e9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.708 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.79d: %ImplicitAs.type.4bd = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.4e9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ba1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.79d) [concrete]
// CHECK:STDOUT:   %.e42: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ba1, %ImplicitAs.facet.79d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.a9c, %Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     .LongLongResult = %LongLongResult.decl
// CHECK:STDOUT:     .PassLongLong = %PassLongLong.cpp_overload_set.value
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .LongLong64 = %Core.LongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.LongLong64: type = import_ref Core//prelude/types/cpp/int, LongLong64, loaded [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.c1f2e9.1: %PassLongLong__carbon_thunk.type.322742.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.ad287c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.c1f2e9.2: %PassLongLong__carbon_thunk.type.322742.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.ad287c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.fa8 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.d2b = impl_witness_table (%Core.import_ref.67b339.2), @Core.IntLiteral.as.As.impl.c99 [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b339.3: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.708 = impl_witness_table (%Core.import_ref.67b339.3), @Cpp.long_long.as.ImplicitAs.impl.82f [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.b3d = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc15_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc15_38.2: %Cpp.long_long = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = value_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.a44 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16_87.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.a89 = addr_of %.loc16_87.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.c1f2e9.1(%cpp_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_87.2: init %LongLongResult to %.loc16_87.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_39: type = splice_block %LongLongResult.ref.loc16 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc16: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_87.3: ref %LongLongResult = temporary %.loc16_87.1, %.loc16_87.2
// CHECK:STDOUT:   %.loc16_87.4: %LongLongResult = acquire_value %.loc16_87.3
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = value_binding cpp_long_long_result, %.loc16_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long.patt: %pattern_type.b3d = value_binding_pattern cpp_compat_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc18: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %LongLong64.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.bd5028.1 [concrete = imports.%CppCompat.bd5028.1]
// CHECK:STDOUT:     %LongLong64.ref: type = name_ref LongLong64, imports.%Core.LongLong64 [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long: %Cpp.long_long = value_binding cpp_compat_long_long, %cpp_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.a44 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc19_101.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.a89 = addr_of %.loc19_101.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.c1f2e9.1(%cpp_compat_long_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_101.2: init %LongLongResult to %.loc19_101.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_46: type = splice_block %LongLongResult.ref.loc19 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc19: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_101.3: ref %LongLongResult = temporary %.loc19_101.1, %.loc19_101.2
// CHECK:STDOUT:   %.loc19_101.4: %LongLongResult = acquire_value %.loc19_101.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = value_binding cpp_compat_long_long_result, %.loc19_101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.c8a = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc21_42: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc21_39.1: %i64 = as_compatible %cpp_long_long.ref.loc21
// CHECK:STDOUT:   %.loc21_39.2: %i64 = converted %cpp_long_long.ref.loc21, %.loc21_39.1
// CHECK:STDOUT:   %i64.loc21_19: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc21_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.5f3 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_77.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.618 = addr_of %.loc22_77.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.c1f2e9.2(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_77.2: init %LongResult to %.loc22_77.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_77.3: ref %LongResult = temporary %.loc22_77.1, %.loc22_77.2
// CHECK:STDOUT:   %.loc22_77.4: %LongResult = acquire_value %.loc22_77.3
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = value_binding carbon_i64_result, %.loc22_77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc24_55.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_55.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_55.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_55.4: init %f32.750 to %.loc24_55.3 = in_place_init %.loc24_55.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_55.5: init %array_type to %a.var = array_init (%.loc24_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_55.1, %.loc24_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.c3e = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.e42 = impl_witness_access constants.%ImplicitAs.impl_witness.4e9, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
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
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f5a: <witness> = impl_witness imports.%Copy.impl_witness_table.1bb [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.f5a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3a3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3a3, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.1bb = impl_witness_table (%Core.import_ref.59e), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.b3d = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.b3d = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_33: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.ce7 = impl_witness_access constants.%Copy.impl_witness.f5a, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_33, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc9(%.loc9_33)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.e7f: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.b6d: <witness> = impl_witness imports.%EqWith.impl_witness_table.b0b [concrete]
// CHECK:STDOUT:   %EqWith.facet.c01: %EqWith.type.e7f = facet_value %Cpp.long_long, (%EqWith.impl_witness.b6d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.830: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.c01) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.1cb: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.c01) [concrete]
// CHECK:STDOUT:   %.29c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.830, %EqWith.facet.c01 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1ca: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.330: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.1cb, %EqWith.facet.c01 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.a6a: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9e2: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.af0: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.376 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.031: %OrderedWith.type.9e2 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.af0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.1a2: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.031) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.f6f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.031) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.b4c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.031) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.da5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.031) [concrete]
// CHECK:STDOUT:   %.c3c: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.b4c, %OrderedWith.facet.031 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.7c8: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.def: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.1a2, %OrderedWith.facet.031 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.606: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.da5, %OrderedWith.facet.031 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.f6f, %OrderedWith.facet.031 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.206: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.b0b = impl_witness_table (%Core.import_ref.74d, %Core.import_ref.206), @Cpp.long_long.as.EqWith.impl.552 [concrete]
// CHECK:STDOUT:   %Core.import_ref.6ae: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.dfe: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.148: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.3cc: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.376 = impl_witness_table (%Core.import_ref.6ae, %Core.import_ref.dfe, %Core.import_ref.148, %Core.import_ref.3cc), @Cpp.long_long.as.OrderedWith.impl.602 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc8_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.b3d = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.29c = impl_witness_access constants.%EqWith.impl_witness.b6d, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.330 = impl_witness_access constants.%EqWith.impl_witness.b6d, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.c3c = impl_witness_access constants.%OrderedWith.impl_witness.af0, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.def = impl_witness_access constants.%OrderedWith.impl_witness.af0, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.70c = impl_witness_access constants.%OrderedWith.impl_witness.af0, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.b47 = impl_witness_access constants.%OrderedWith.impl_witness.af0, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.521: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.4a2: %ImplicitAs.type.6ca = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.f89(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.ec3b42.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.f89(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.ec3b42.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.f89(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.f55f98.1: %Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.f89(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.f55f98.2: %Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.290: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8e2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.34a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.50a: %ImplicitAs.type.6ca = facet_value %i64, (%ImplicitAs.impl_witness.8e2) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.5ad: <witness> = impl_witness imports.%EqWith.impl_witness_table.9f3, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.d1d16e.1: %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.d1d16e.2: %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.54c: %EqWith.type.521 = facet_value %Cpp.long_long, (%EqWith.impl_witness.5ad) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.2f0: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.54c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.e61: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.54c) [concrete]
// CHECK:STDOUT:   %.e6a: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.2f0, %EqWith.facet.54c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a92af9.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.d1d16e.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.99b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.99e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.99b, %ImplicitAs.facet.50a [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a92af9.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.d1d16e.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.82b: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.e61, %EqWith.facet.54c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.551592.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.551592.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.fb5: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4bbe1d.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4bbe1d.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.3fabfe.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.3fabfe.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8e72eb.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8e72eb.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b018d7.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%T.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.b018d7.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.d0c: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.82c: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.02e, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.3b128e.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.f82727.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.3b128e.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.f82727.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.cab: %OrderedWith.type.fb5 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.82c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b0b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.1ce: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9d8: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.3dd: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %.2e1: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9d8, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.6b3581.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.f82727.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.6b3581.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.f82727.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.6a4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b0b, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.30d7b9.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.3b128e.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.30d7b9.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.3b128e.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.c4e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.3dd, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.186cf8.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.186cf8.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.5ec: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.1ce, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.6e2d96.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.6e2d96.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.7b2: <witness> = impl_witness imports.%EqWith.impl_witness_table.9f3, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a6f59e.1: %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a6f59e.2: %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.f89(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.207: %EqWith.type.290 = facet_value %Cpp.long_long, (%EqWith.impl_witness.7b2) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.a85: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.207) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.3cf: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.207) [concrete]
// CHECK:STDOUT:   %.4ab: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.a85, %EqWith.facet.207 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a8940a.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a6f59e.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a8940a.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a6f59e.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.144: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.3cf, %EqWith.facet.207 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.24481d.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.24481d.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.6c1: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.02e, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.f1f076.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.f1f076.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.dfc(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.9bd: %OrderedWith.type.d0c = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.6c1) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.184: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.9bd) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.6c4: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.9bd) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d1d: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.9bd) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.c60: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.9bd) [concrete]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d1d, %OrderedWith.facet.9bd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.2a25d1.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.2a25d1.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.3c8: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.184, %OrderedWith.facet.9bd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.4c149a.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.f1f076.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.4c149a.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.f1f076.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.7e2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.c60, %OrderedWith.facet.9bd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.639629.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.639629.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.6c4, %OrderedWith.facet.9bd [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.317dcc.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.317dcc.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.e71: @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.f55f98.1)]
// CHECK:STDOUT:   %Core.import_ref.01e: @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.ec3b42.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.9f3 = impl_witness_table (%Core.import_ref.e71, %Core.import_ref.01e), @Cpp.long_long.as.EqWith.impl.f89 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.806: @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a6df99.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.ec3b42.2)]
// CHECK:STDOUT:   %Core.Equal.d2e: @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.1ac185.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.f89.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.f55f98.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.34a = impl_witness_table (%Core.import_ref.67b339.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.54f: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.b018d7.1)]
// CHECK:STDOUT:   %Core.import_ref.9ee: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8e72eb.1)]
// CHECK:STDOUT:   %Core.import_ref.cd0: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.3fabfe.1)]
// CHECK:STDOUT:   %Core.import_ref.383: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4bbe1d.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.02e = impl_witness_table (%Core.import_ref.54f, %Core.import_ref.9ee, %Core.import_ref.cd0, %Core.import_ref.383), @Cpp.long_long.as.OrderedWith.impl.dfc [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.d58: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.d017bf.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4bbe1d.2)]
// CHECK:STDOUT:   %Core.Greater.6ef: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.84c905.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.3fabfe.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.bff: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7d1c25.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8e72eb.2)]
// CHECK:STDOUT:   %Core.Less.f3d: @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.b51657.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.dfc.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.b018d7.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.e6a = impl_witness_access constants.%EqWith.impl_witness.5ad, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.d1d16e.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a92af9.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.1 = specific_constant imports.%Core.Equal.d2e, @Cpp.long_long.as.EqWith.impl.f89(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.d1d16e.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.87351e.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.d1d16e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a92af9.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.82b = impl_witness_access constants.%EqWith.impl_witness.5ad, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.551592.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.1 = specific_constant imports.%Core.NotEqual.806, @Cpp.long_long.as.EqWith.impl.f89(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.14dd7e.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4c3a96.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.551592.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.2e1 = impl_witness_access constants.%OrderedWith.impl_witness.82c, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.f82727.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.6b3581.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.1 = specific_constant imports.%Core.Greater.6ef, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.f82727.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f12403.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.f82727.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.6b3581.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.6a4 = impl_witness_access constants.%OrderedWith.impl_witness.82c, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.3b128e.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.30d7b9.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.1 = specific_constant imports.%Core.Less.f3d, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.3b128e.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.9f52a3.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.3b128e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.30d7b9.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.c4e = impl_witness_access constants.%OrderedWith.impl_witness.82c, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.186cf8.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.1 = specific_constant imports.%Core.GreaterOrEquivalent.d58, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1c2e4c.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.7b0705.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.186cf8.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.5ec = impl_witness_access constants.%OrderedWith.impl_witness.82c, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.6e2d96.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.1 = specific_constant imports.%Core.LessOrEquivalent.bff, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3446a2.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d359a2.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long_long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long_long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.6e2d96.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.4ab = impl_witness_access constants.%EqWith.impl_witness.7b2, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a6f59e.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a8940a.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.1 = specific_constant imports.%Core.Equal.d2e, @Cpp.long_long.as.EqWith.impl.f89(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a6f59e.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.831cf8.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a6f59e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.a8940a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.144 = impl_witness_access constants.%EqWith.impl_witness.7b2, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.24481d.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.1 = specific_constant imports.%Core.NotEqual.806, @Cpp.long_long.as.EqWith.impl.f89(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.17f59f.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.9f4b2f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.24481d.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.994 = impl_witness_access constants.%OrderedWith.impl_witness.6c1, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.2a25d1.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.1 = specific_constant imports.%Core.Greater.6ef, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.c196a9.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.e6322a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.2a25d1.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.3c8 = impl_witness_access constants.%OrderedWith.impl_witness.6c1, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.f1f076.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.4c149a.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.1 = specific_constant imports.%Core.Less.f3d, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.f1f076.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.c2ca83.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.f1f076.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.4c149a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.7e2 = impl_witness_access constants.%OrderedWith.impl_witness.6c1, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.639629.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.1 = specific_constant imports.%Core.GreaterOrEquivalent.d58, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.46c950.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2ea354.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long_long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long_long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.639629.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.d6b = impl_witness_access constants.%OrderedWith.impl_witness.6c1, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.03c]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.317dcc.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.1 = specific_constant imports.%Core.LessOrEquivalent.bff, @Cpp.long_long.as.OrderedWith.impl.dfc(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.477d8c.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.64175b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long_long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long_long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.03c) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.317dcc.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %Cpp.long_long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.a9c]
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.e7f: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.4a2: %ImplicitAs.type.6ca = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.36a(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.9ed3da.1: %T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.36a(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.9ed3da.2: %T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.bd9364.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.36a(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.028cd2.1: %T.as_type.as.EqWith.impl.Equal.type.bd9364.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.bd9364.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.36a(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.028cd2.2: %T.as_type.as.EqWith.impl.Equal.type.bd9364.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8e2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.34a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.50a: %ImplicitAs.type.6ca = facet_value %i64, (%ImplicitAs.impl_witness.8e2) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f90: <witness> = impl_witness imports.%EqWith.impl_witness_table.c31, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.5d905d.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.12b65c.1: %T.as_type.as.EqWith.impl.Equal.type.5d905d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.2318fe.1: %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.5d905d.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.12b65c.2: %T.as_type.as.EqWith.impl.Equal.type.5d905d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.2318fe.2: %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.50b: %EqWith.type.e7f = facet_value %i64, (%EqWith.impl_witness.f90) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.bf8: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.50b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.9d1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.50b) [concrete]
// CHECK:STDOUT:   %.3a7: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.bf8, %EqWith.facet.50b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.42ba9d.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.12b65c.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.42ba9d.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.12b65c.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.99b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.99e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.99b, %ImplicitAs.facet.50a [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.2e2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.9d1, %EqWith.facet.50b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.b0defc.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.2318fe.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.b0defc.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.2318fe.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9e2: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.52c5b7.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.52c5b7.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.911cf7.1: %T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.911cf7.2: %T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a9cbdb.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a9cbdb.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.766515.1: %T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.246(%T.4a2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.766515.2: %T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.a87: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.c87, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.98f4cb.1: %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.3eef78.1: %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.98f4cb.2: %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.3eef78.2: %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.ac3: %OrderedWith.type.9e2 = facet_value %i64, (%OrderedWith.impl_witness.a87) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c28: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.ac3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.10e: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.ac3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.92a: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.ac3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7e6: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.ac3) [concrete]
// CHECK:STDOUT:   %.162: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.92a, %OrderedWith.facet.ac3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.460413.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.3eef78.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.460413.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.3eef78.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.eb1: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c28, %OrderedWith.facet.ac3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.530c39.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.98f4cb.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.530c39.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.98f4cb.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.9d3: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7e6, %OrderedWith.facet.ac3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b8d742.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b8d742.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.829: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.10e, %OrderedWith.facet.ac3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a86854.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a86854.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f83: <witness> = impl_witness imports.%EqWith.impl_witness_table.c31, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.42aa9c.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c862ed.1: %T.as_type.as.EqWith.impl.Equal.type.42aa9c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.cdca77.1: %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.42aa9c.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c862ed.2: %T.as_type.as.EqWith.impl.Equal.type.42aa9c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.36a(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.cdca77.2: %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.849: %EqWith.type.e7f = facet_value Core.IntLiteral, (%EqWith.impl_witness.f83) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b75: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.849) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.1bd: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.849) [concrete]
// CHECK:STDOUT:   %.2fb: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b75, %EqWith.facet.849 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.74f389.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.c862ed.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c862ed.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3d463b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.74f389.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.c862ed.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c862ed.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3d463b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.2 [concrete]
// CHECK:STDOUT:   %.ea1: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.1bd, %EqWith.facet.849 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.4cf479.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.cdca77.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.cdca77.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3588b9.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.4cf479.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.cdca77.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.cdca77.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3588b9.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.758: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.c87, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.5779c8.1: %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.23750b.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.846662.1: %T.as_type.as.OrderedWith.impl.Greater.type.23750b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.5779c8.2: %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.23750b.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.846662.2: %T.as_type.as.OrderedWith.impl.Greater.type.23750b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.246(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.456: %OrderedWith.type.9e2 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.758) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.671: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.456) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.816: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.456) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.643: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.456) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.53a: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.456) [concrete]
// CHECK:STDOUT:   %.81e: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.643, %OrderedWith.facet.456 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.6a8fe5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.846662.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.846662.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.4d15cb.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.6a8fe5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.846662.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.846662.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.4d15cb.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.2 [concrete]
// CHECK:STDOUT:   %.876: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.671, %OrderedWith.facet.456 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.c60eb5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.5779c8.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.5779c8.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.ecbb78.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.c60eb5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.5779c8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.5779c8.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.ecbb78.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.2 [concrete]
// CHECK:STDOUT:   %.213: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.53a, %OrderedWith.facet.456 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b446f9.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.e58891.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b446f9.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.e58891.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.2 [concrete]
// CHECK:STDOUT:   %.4a5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.816, %OrderedWith.facet.456 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0dffd2.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3e4a5f.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0dffd2.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %bound_method.3e4a5f.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.124: @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.bd9364.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.028cd2.1)]
// CHECK:STDOUT:   %Core.import_ref.c93: @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.9ed3da.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.c31 = impl_witness_table (%Core.import_ref.124, %Core.import_ref.c93), @T.as_type.as.EqWith.impl.36a [concrete]
// CHECK:STDOUT:   %Core.NotEqual.3b2: @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.e0fa64.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.9ed3da.2)]
// CHECK:STDOUT:   %Core.Equal.2a6: @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.bd9364.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.36a.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.028cd2.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.34a = impl_witness_table (%Core.import_ref.67b339.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.869: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.766515.1)]
// CHECK:STDOUT:   %Core.import_ref.a12: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.a9cbdb.1)]
// CHECK:STDOUT:   %Core.import_ref.6a7: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.911cf7.1)]
// CHECK:STDOUT:   %Core.import_ref.567: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.52c5b7.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.c87 = impl_witness_table (%Core.import_ref.869, %Core.import_ref.a12, %Core.import_ref.6a7, %Core.import_ref.567), @T.as_type.as.OrderedWith.impl.246 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.794: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.ccc470.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.52c5b7.2)]
// CHECK:STDOUT:   %Core.Greater.607: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.c6a2b3.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.911cf7.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.eec: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d4a344.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.a9cbdb.2)]
// CHECK:STDOUT:   %Core.Less.edb: @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.1ea1c7.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.246.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.766515.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.3a7 = impl_witness_access constants.%EqWith.impl_witness.f90, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.12b65c.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.42ba9d.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.5d905d.1 = specific_constant imports.%Core.Equal.2a6, @T.as_type.as.EqWith.impl.36a(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.12b65c.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.5d905d.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.12b65c.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.42ba9d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.2e2 = impl_witness_access constants.%EqWith.impl_witness.f90, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.2318fe.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.b0defc.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.1 = specific_constant imports.%Core.NotEqual.3b2, @T.as_type.as.EqWith.impl.36a(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.2318fe.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.3c06f1.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.2318fe.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.b0defc.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.162 = impl_witness_access constants.%OrderedWith.impl_witness.a87, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.3eef78.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.460413.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.1 = specific_constant imports.%Core.Greater.607, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.3eef78.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.92cd87.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.3eef78.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.460413.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.eb1 = impl_witness_access constants.%OrderedWith.impl_witness.a87, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.98f4cb.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.530c39.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.1 = specific_constant imports.%Core.Less.edb, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.98f4cb.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.2caa1b.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.98f4cb.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.530c39.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.9d3 = impl_witness_access constants.%OrderedWith.impl_witness.a87, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b8d742.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.1 = specific_constant imports.%Core.GreaterOrEquivalent.794, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.5c7593.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.b86b9e.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b8d742.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.829 = impl_witness_access constants.%OrderedWith.impl_witness.a87, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a86854.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.1 = specific_constant imports.%Core.LessOrEquivalent.eec, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f07c8e.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.7b7d5b.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.50a) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a86854.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.2fb = impl_witness_access constants.%EqWith.impl_witness.f83, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c862ed.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.74f389.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.3d463b.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.42aa9c.1 = specific_constant imports.%Core.Equal.2a6, @T.as_type.as.EqWith.impl.36a(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c862ed.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.42aa9c.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c862ed.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.74f389.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.a814b0.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.3d463b.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.ea1 = impl_witness_access constants.%EqWith.impl_witness.f83, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cdca77.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.4cf479.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.3588b9.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.1 = specific_constant imports.%Core.NotEqual.3b2, @T.as_type.as.EqWith.impl.36a(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cdca77.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.6155f0.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cdca77.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.4cf479.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ec27aa.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.3588b9.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.81e = impl_witness_access constants.%OrderedWith.impl_witness.758, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.846662.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.6a8fe5.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.4d15cb.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.23750b.1 = specific_constant imports.%Core.Greater.607, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.846662.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.23750b.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.846662.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.6a8fe5.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.815a8f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.4d15cb.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.876 = impl_witness_access constants.%OrderedWith.impl_witness.758, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5779c8.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.c60eb5.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.ecbb78.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.1 = specific_constant imports.%Core.Less.edb, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5779c8.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.ab38e0.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5779c8.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.c60eb5.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.934373.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.ecbb78.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.213 = impl_witness_access constants.%OrderedWith.impl_witness.758, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b446f9.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.e58891.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.1 = specific_constant imports.%Core.GreaterOrEquivalent.794, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9894aa.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7f8c13.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b446f9.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.072b05.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.e58891.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.4a5 = impl_witness_access constants.%OrderedWith.impl_witness.758, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0dffd2.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.3e4a5f.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.1 = specific_constant imports.%Core.LessOrEquivalent.eec, @T.as_type.as.OrderedWith.impl.246(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cc1463.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3fee12.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0dffd2.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.03c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.69f783.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.3e4a5f.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long_long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.a9c]
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.b99: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.228) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.56c: %T.as_type.as.ImplicitAs.impl.Convert.type.b99 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ae6: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.4a2: %ImplicitAs.type.6ca = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f5a: <witness> = impl_witness imports.%Copy.impl_witness_table.1bb [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.f5a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.eef, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.77f: %ImplicitAs.type.6ca = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.5a1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.f14: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.04a, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.847: %AddAssignWith.type.ae6 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.f14) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.a4d: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet.847) [concrete]
// CHECK:STDOUT:   %.c8e: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.a4d, %AddAssignWith.facet.847 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fdd3c9.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fdd3c9.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.920: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.cc6: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.bb1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.e15: %SubAssignWith.type.920 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.cc6) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.a03: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet.e15) [concrete]
// CHECK:STDOUT:   %.128: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.a03, %SubAssignWith.facet.e15 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.df6869.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.df6869.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.28b: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.d83: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.1c9, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d68322.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d68322.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.757: %MulAssignWith.type.28b = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.d83) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5eb: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet.757) [concrete]
// CHECK:STDOUT:   %.df3: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.5eb, %MulAssignWith.facet.757 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.74bb28.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.d68322.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.74bb28.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.d68322.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.5c9: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.749c34.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.749c34.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.793: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.8e6, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.90de40.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.90de40.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.8ea: %DivAssignWith.type.5c9 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.793) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.9a1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet.8ea) [concrete]
// CHECK:STDOUT:   %.dd8: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.9a1, %DivAssignWith.facet.8ea [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.ee6bc2.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.90de40.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.ee6bc2.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.90de40.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.064: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.146: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.8cd, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.e01: %ModAssignWith.type.064 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.146) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.d4d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet.e01) [concrete]
// CHECK:STDOUT:   %.ee4: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.d4d, %ModAssignWith.facet.e01 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.910fb0.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.910fb0.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.719: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.796: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.a9f, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.41f: %BitAndAssignWith.type.719 = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.796) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.9e3: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet.41f) [concrete]
// CHECK:STDOUT:   %.eb6: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.9e3, %BitAndAssignWith.facet.41f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99004a.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99004a.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.bac: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.66f: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.f01, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.a3c: %BitOrAssignWith.type.bac = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.66f) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.f50: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet.a3c) [concrete]
// CHECK:STDOUT:   %.1be: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.f50, %BitOrAssignWith.facet.a3c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.f33a3e.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.f33a3e.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.05a: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.d23: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.755, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.822: %BitXorAssignWith.type.05a = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.d23) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.404: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet.822) [concrete]
// CHECK:STDOUT:   %.30b: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.404, %BitXorAssignWith.facet.822 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adecc.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adecc.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.8e3: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.13c: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.997, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.f51: %LeftShiftAssignWith.type.8e3 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.13c) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.a22: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet.f51) [concrete]
// CHECK:STDOUT:   %.047: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.a22, %LeftShiftAssignWith.facet.f51 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.0338e3.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.0338e3.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.de1: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.b5b: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.794, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.de6: %RightShiftAssignWith.type.de1 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.b5b) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.1bd: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet.de6) [concrete]
// CHECK:STDOUT:   %.949: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.1bd, %RightShiftAssignWith.facet.de6 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.6c8346.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.6c8346.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.77f) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5e1: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.b99) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.56c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.eef = impl_witness_table (%Core.import_ref.5e1), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.b55: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.04a = impl_witness_table (%Core.import_ref.b55), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c7: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2)]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.1bb = impl_witness_table (%Core.import_ref.59e), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f1b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.bb1 = impl_witness_table (%Core.import_ref.f1b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.387: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.2)]
// CHECK:STDOUT:   %Core.import_ref.4af: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.1c9 = impl_witness_table (%Core.import_ref.4af), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.211: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.2)]
// CHECK:STDOUT:   %Core.import_ref.ae0: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.749c34.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.8e6 = impl_witness_table (%Core.import_ref.ae0), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.62a: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.749c34.2)]
// CHECK:STDOUT:   %Core.import_ref.847: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.8cd = impl_witness_table (%Core.import_ref.847), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.853: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.2)]
// CHECK:STDOUT:   %Core.import_ref.ff5: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.a9f = impl_witness_table (%Core.import_ref.ff5), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dcb: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.2)]
// CHECK:STDOUT:   %Core.import_ref.f34: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.f01 = impl_witness_table (%Core.import_ref.f34), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.701: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.2)]
// CHECK:STDOUT:   %Core.import_ref.d5f: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.755 = impl_witness_table (%Core.import_ref.d5f), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d9c: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.2)]
// CHECK:STDOUT:   %Core.import_ref.2b2: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.997 = impl_witness_table (%Core.import_ref.2b2), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aed: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.2)]
// CHECK:STDOUT:   %Core.import_ref.47a: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.794 = impl_witness_table (%Core.import_ref.47a), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.b3d = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.b3d = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.c8e = impl_witness_access constants.%AddAssignWith.impl_witness.f14, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fdd3c9.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.1 = specific_constant imports.%Core.Op.7c7, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.29ca7a.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7efc37.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fdd3c9.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.128 = impl_witness_access constants.%SubAssignWith.impl_witness.cc6, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.df6869.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.1 = specific_constant imports.%Core.Op.387, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.843e57.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.4e6c84.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.df6869.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.df3 = impl_witness_access constants.%MulAssignWith.impl_witness.d83, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d68322.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.74bb28.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.1 = specific_constant imports.%Core.Op.211, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d68322.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.7fe7c8.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d68322.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.74bb28.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.dd8 = impl_witness_access constants.%DivAssignWith.impl_witness.793, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.90de40.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.ee6bc2.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.1 = specific_constant imports.%Core.Op.62a, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.90de40.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.aa6ac5.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.90de40.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.ee6bc2.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.ee4 = impl_witness_access constants.%ModAssignWith.impl_witness.146, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.910fb0.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.1 = specific_constant imports.%Core.Op.853, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1ed032.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.2792fd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.910fb0.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.eb6 = impl_witness_access constants.%BitAndAssignWith.impl_witness.796, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99004a.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.1 = specific_constant imports.%Core.Op.dcb, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.44d9c1.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.156e35.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99004a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.1be = impl_witness_access constants.%BitOrAssignWith.impl_witness.66f, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.f33a3e.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.1 = specific_constant imports.%Core.Op.701, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.58f268.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.5dca66.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.f33a3e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.30b = impl_witness_access constants.%BitXorAssignWith.impl_witness.d23, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adecc.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.1 = specific_constant imports.%Core.Op.d9c, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.b0e4ed.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ae0dd3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adecc.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.047 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.13c, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.0338e3.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.1 = specific_constant imports.%Core.Op.aed, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.f49131.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a4706f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.0338e3.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.949 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.b5b, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.6ca = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.5a1) [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.6ca = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.77f]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.6c8346.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.1 = specific_constant imports.%Core.Op.c23, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.43e888.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.155553.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.77f) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.6c8346.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.1bd: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.4a2: %ImplicitAs.type.6ca = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8e2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.34a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.50a: %ImplicitAs.type.6ca = facet_value %i64, (%ImplicitAs.impl_witness.8e2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.133: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.04a, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.eb5: %AddAssignWith.type.1bd = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.133) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.c3b: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.eb5) [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.c3b, %AddAssignWith.facet.eb5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.99b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.99e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.99b, %ImplicitAs.facet.50a [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.392: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.8b9: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.bb1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.6a8: %SubAssignWith.type.392 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.8b9) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.f63: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet.6a8) [concrete]
// CHECK:STDOUT:   %.7cb: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.f63, %SubAssignWith.facet.6a8 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7536ce.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7536ce.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.aba: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.97a: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.1c9, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.a19: %MulAssignWith.type.aba = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.97a) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.0a4: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet.a19) [concrete]
// CHECK:STDOUT:   %.911: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.0a4, %MulAssignWith.facet.a19 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.e70587.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.e70587.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.ddd: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.749c34.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.749c34.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.506: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.8e6, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.757978.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.757978.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.9e0: %DivAssignWith.type.ddd = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.506) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.03c: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet.9e0) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.03c, %DivAssignWith.facet.9e0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6c7721.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.757978.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6c7721.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.757978.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.f34: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.c40: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.8cd, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.152e13.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.152e13.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.942: %ModAssignWith.type.f34 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.f72: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet.942) [concrete]
// CHECK:STDOUT:   %.4a2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.f72, %ModAssignWith.facet.942 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.fe4355.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.152e13.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.fe4355.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.152e13.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.bdd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.e54: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.a9f, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.c91: %BitAndAssignWith.type.bdd = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.e54) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.99d: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet.c91) [concrete]
// CHECK:STDOUT:   %.2a5: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.99d, %BitAndAssignWith.facet.c91 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.705501.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.705501.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.9a3: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.9bda: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.f01, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.8f6: %BitOrAssignWith.type.9a3 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.9bda) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.2cd: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet.8f6) [concrete]
// CHECK:STDOUT:   %.798: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.2cd, %BitOrAssignWith.facet.8f6 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.aa5799.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.aa5799.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.b1f: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.ae5: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.755, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.434: %BitXorAssignWith.type.b1f = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.ae5) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.d5d: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet.434) [concrete]
// CHECK:STDOUT:   %.b19: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.d5d, %BitXorAssignWith.facet.434 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adcc6.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adcc6.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.311: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.45b: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.997, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.28c: %LeftShiftAssignWith.type.311 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.45b) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.c83: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet.28c) [concrete]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.c83, %LeftShiftAssignWith.facet.28c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.d155d4.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.d155d4.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.311: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.39d: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.794, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.74b: %RightShiftAssignWith.type.311 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.39d) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.335: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet.74b) [concrete]
// CHECK:STDOUT:   %.3a3: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.335, %RightShiftAssignWith.facet.74b [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.811e22.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.811e22.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.b554: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.04a = impl_witness_table (%Core.import_ref.b554), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c7: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.34a = impl_witness_table (%Core.import_ref.67b339.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f1b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.bb1 = impl_witness_table (%Core.import_ref.f1b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.387: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.a010c9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2bb15c.2)]
// CHECK:STDOUT:   %Core.import_ref.4af: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.1c9 = impl_witness_table (%Core.import_ref.4af), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.211: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.e851ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.51e84d.2)]
// CHECK:STDOUT:   %Core.import_ref.ae0: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.749c34.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.8e6 = impl_witness_table (%Core.import_ref.ae0), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.62a: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.b19641.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.749c34.2)]
// CHECK:STDOUT:   %Core.import_ref.847: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.8cd = impl_witness_table (%Core.import_ref.847), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.853: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d8f7e2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.ba402b.2)]
// CHECK:STDOUT:   %Core.import_ref.ff5: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.a9f = impl_witness_table (%Core.import_ref.ff5), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dcb: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.77ab0f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.08ecb3.2)]
// CHECK:STDOUT:   %Core.import_ref.f34: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.f01 = impl_witness_table (%Core.import_ref.f34), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.701: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.148240.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.a3f40c.2)]
// CHECK:STDOUT:   %Core.import_ref.d5f: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.755 = impl_witness_table (%Core.import_ref.d5f), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d9c: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.5dce58.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f72ab0.2)]
// CHECK:STDOUT:   %Core.import_ref.2b2: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.997 = impl_witness_table (%Core.import_ref.2b2), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aed: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.593773.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a80a5e.2)]
// CHECK:STDOUT:   %Core.import_ref.47a: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.794 = impl_witness_table (%Core.import_ref.47a), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.66e865.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.ac83e7.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.c0a = impl_witness_access constants.%AddAssignWith.impl_witness.133, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = specific_constant imports.%Core.Op.7c7, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.7cb = impl_witness_access constants.%SubAssignWith.impl_witness.8b9, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7536ce.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.1 = specific_constant imports.%Core.Op.387, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.582b5a.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdfcfb.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7536ce.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.911 = impl_witness_access constants.%MulAssignWith.impl_witness.97a, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.e70587.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.1 = specific_constant imports.%Core.Op.211, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.58953a.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e96eb4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.e70587.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.dca = impl_witness_access constants.%DivAssignWith.impl_witness.506, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.757978.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6c7721.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.1 = specific_constant imports.%Core.Op.62a, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.757978.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.ccff2f.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.757978.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6c7721.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.4a2 = impl_witness_access constants.%ModAssignWith.impl_witness.c40, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.152e13.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.fe4355.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.1 = specific_constant imports.%Core.Op.853, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.152e13.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.4308ce.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.152e13.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.fe4355.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.2a5 = impl_witness_access constants.%BitAndAssignWith.impl_witness.e54, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.705501.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.1 = specific_constant imports.%Core.Op.dcb, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9611ee.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.4820f1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long_long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long_long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.705501.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.798 = impl_witness_access constants.%BitOrAssignWith.impl_witness.9bda, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.aa5799.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.1 = specific_constant imports.%Core.Op.701, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.dc7f7e.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.01b4ec.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.aa5799.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.b19 = impl_witness_access constants.%BitXorAssignWith.impl_witness.ae5, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adcc6.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.1 = specific_constant imports.%Core.Op.d9c, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.0437a7.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.8ab63b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8adcc6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.b08 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.45b, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.d155d4.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.1 = specific_constant imports.%Core.Op.aed, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.7214a5.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.a6ceec.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.d155d4.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.3a3 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.39d, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.811e22.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.1 = specific_constant imports.%Core.Op.c23, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.bcc009.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.bc1495.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.811e22.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20_9: init %Cpp.long_long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20_9
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long_long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.17a: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.4a2: %ImplicitAs.type.6ca = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.afe: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.04a, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.351: %AddAssignWith.type.17a = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.afe) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.2c7: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.351) [concrete]
// CHECK:STDOUT:   %.b26: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.2c7, %AddAssignWith.facet.351 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c697e7.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c697e7.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.b55: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.04a = impl_witness_table (%Core.import_ref.b55), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c7: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.b26 = impl_witness_access constants.%AddAssignWith.impl_witness.afe, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f04) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.6ca = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c697e7.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.1 = specific_constant imports.%Core.Op.7c7, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c02bd2.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8ab203.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long_long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long_long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c697e7.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8: init %Cpp.long_long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.4(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e82: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03c: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.03c) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet.03c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0da(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a33: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1be, @Core.IntLiteral.as.ImplicitAs.impl.0da(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d03: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0da(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b11: %ImplicitAs.type.e82 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a33) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.486: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.b11) [concrete]
// CHECK:STDOUT:   %.036: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.486, %ImplicitAs.facet.b11 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.93d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.9e7: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.1bd: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.4a2: %ImplicitAs.type.6ca = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.4a2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8e2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.34a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.50a: %ImplicitAs.type.6ca = facet_value %i64, (%ImplicitAs.impl_witness.8e2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.133: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.04a, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.eb5: %AddAssignWith.type.1bd = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.133) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.c3b: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.eb5) [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.c3b, %AddAssignWith.facet.eb5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.99b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %.99e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.99b, %ImplicitAs.facet.50a [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.50a) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f3: @Core.IntLiteral.as.ImplicitAs.impl.0da.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0da.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1be = impl_witness_table (%Core.import_ref.7f3), @Core.IntLiteral.as.ImplicitAs.impl.0da [concrete]
// CHECK:STDOUT:   %Core.import_ref.b55: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.04a = impl_witness_table (%Core.import_ref.b55), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c7: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.ee41a4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.e9d8f9.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.34a = impl_witness_table (%Core.import_ref.67b339.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c8a = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc9: %.036 = impl_witness_access constants.%ImplicitAs.impl_witness.a33, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.93d]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.c0a = impl_witness_access constants.%AddAssignWith.impl_witness.133, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.8e2) [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6ca = converted constants.%i64, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.50a]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = specific_constant imports.%Core.Op.7c7, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.c22ca3.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.b8c040.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.50a) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.ed88bd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.99e = impl_witness_access constants.%ImplicitAs.impl_witness.8e2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.b3d: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ca: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f04: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.310 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6ca = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f04) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e94: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.710: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e94, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.a9c: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.3cd: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.9bf: type = fn_type_with_self_type %Inc.WithSelf.Op.type.3cd, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.656: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.3b4: type = fn_type_with_self_type %Dec.WithSelf.Op.type.656, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.type: type = fn_type @Cpp.long_long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op: %Cpp.long_long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.310 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.f9d [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e3cf.1: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.59e3cf.1), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e3cf.2: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.59e3cf.2), @Cpp.long_long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b3d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.710 = impl_witness_access constants.%ImplicitAs.impl_witness.f04, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.a9c]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.9bf = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.3b4 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.458: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.788: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.704: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.71b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d99 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.72f: %ImplicitAs.type.788 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.71b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f14: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.72f) [concrete]
// CHECK:STDOUT:   %.693: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f14, %ImplicitAs.facet.72f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b67: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.30b: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.089: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.5cbe77.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.fb9213.1: %PassULongLong__carbon_thunk.type.5cbe77.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bb8: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.093: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c97, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.de8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.de8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.338: %ImplicitAs.type.704 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.093) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.25c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.338) [concrete]
// CHECK:STDOUT:   %.f05: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.25c, %ImplicitAs.facet.338 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b89: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.818: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.21a: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %ptr.cd3: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.5cbe77.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.fb9213.2: %PassULongLong__carbon_thunk.type.5cbe77.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.fa8: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.122: type = facet_type <@As, @As(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.05b: <witness> = impl_witness imports.%As.impl_witness_table.fc4 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.122 = facet_value Core.IntLiteral, (%As.impl_witness.05b) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.c8f: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.5d2: type = fn_type_with_self_type %As.WithSelf.Convert.type.c8f, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.800: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f54 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.27c: %ImplicitAs.type.4bd = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.800) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.068: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.27c) [concrete]
// CHECK:STDOUT:   %.259: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.068, %ImplicitAs.facet.27c [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.b67, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     .ULongLongResult = %ULongLongResult.decl
// CHECK:STDOUT:     .PassULongLong = %PassULongLong.cpp_overload_set.value
// CHECK:STDOUT:     .ULongResult = %ULongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULongLong64 = %Core.ULongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULongLong64: type = import_ref Core//prelude/types/cpp/int, ULongLong64, loaded [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d99 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.5cf [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.1b9bd8.1: %PassULongLong__carbon_thunk.type.5cbe77.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.fb9213.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.0ae: @Core.IntLiteral.as.ImplicitAs.impl.f1e.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.f1e.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c97 = impl_witness_table (%Core.import_ref.0ae), @Core.IntLiteral.as.ImplicitAs.impl.f1e [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.1b9bd8.2: %PassULongLong__carbon_thunk.type.5cbe77.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.fb9213.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.fa8 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.fc4 = impl_witness_table (%Core.import_ref.67b339.2), @Core.IntLiteral.as.As.impl.869 [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b339.3: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f54 = impl_witness_table (%Core.import_ref.67b339.3), @Cpp.unsigned_long_long.as.ImplicitAs.impl.816 [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.458 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc15: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.693 = impl_witness_access constants.%ImplicitAs.impl_witness.71b, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.b67]
// CHECK:STDOUT:   %.loc15_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.b67]
// CHECK:STDOUT:   %.loc15_56.2: %Cpp.unsigned_long_long = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.b67]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = value_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.30b = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc16: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.089 = addr_of %.loc16_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.1b9bd8.1(%cpp_unsigned_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_107.2: init %ULongLongResult to %.loc16_107.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_48: type = splice_block %ULongLongResult.ref.loc16 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc16: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_107.3: ref %ULongLongResult = temporary %.loc16_107.1, %.loc16_107.2
// CHECK:STDOUT:   %.loc16_107.4: %ULongLongResult = acquire_value %.loc16_107.3
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = value_binding cpp_unsigned_long_long_result, %.loc16_107.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long.patt: %pattern_type.458 = value_binding_pattern cpp_compat_ulong_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc18: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULongLong64.ref [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.bd5028.1 [concrete = imports.%CppCompat.bd5028.1]
// CHECK:STDOUT:     %ULongLong64.ref: type = name_ref ULongLong64, imports.%Core.ULongLong64 [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long: %Cpp.unsigned_long_long = value_binding cpp_compat_ulong_long, %cpp_unsigned_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long_result.patt: %pattern_type.30b = value_binding_pattern cpp_compat_ulong_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc19_105.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.089 = addr_of %.loc19_105.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.1b9bd8.1(%cpp_compat_ulong_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_105.2: init %ULongLongResult to %.loc19_105.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_47: type = splice_block %ULongLongResult.ref.loc19 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc19: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_105.3: ref %ULongLongResult = temporary %.loc19_105.1, %.loc19_105.2
// CHECK:STDOUT:   %.loc19_105.4: %ULongLongResult = acquire_value %.loc19_105.3
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = value_binding cpp_compat_ulong_long_result, %.loc19_105.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.bb8 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc21: %.f05 = impl_witness_access constants.%ImplicitAs.impl_witness.093, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b89]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.818]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %carbon_u64: %u64 = value_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.423 = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_79.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.cd3 = addr_of %.loc22_79.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.1b9bd8.2(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_79.2: init %ULongResult to %.loc22_79.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_79.3: ref %ULongResult = temporary %.loc22_79.1, %.loc22_79.2
// CHECK:STDOUT:   %.loc22_79.4: %ULongResult = acquire_value %.loc22_79.3
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = value_binding carbon_u64_result, %.loc22_79.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc24_64.1: <bound method> = bound_method %float, %impl.elem0.loc24_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_64.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc24_64.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_64.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_64.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_64.4: init %f32.750 to %.loc24_64.3 = in_place_init %.loc24_64.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_64.5: init %array_type to %a.var = array_init (%.loc24_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_64.1, %.loc24_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc24: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.5d2 = impl_witness_access constants.%As.impl_witness.05b, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.b67]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.b67]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.b67]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.259 = impl_witness_access constants.%ImplicitAs.impl_witness.800, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
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
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.458: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.788: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.71b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d99 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.788 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.71b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f14: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.693: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f14, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b67: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f48: <witness> = impl_witness imports.%Copy.impl_witness_table.536 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.f48) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.30b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.dae: type = fn_type_with_self_type %Copy.WithSelf.Op.type.30b, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d99 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.5cf [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.536 = impl_witness_table (%Core.import_ref.59e), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.458 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.458 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.693 = impl_witness_access constants.%ImplicitAs.impl_witness.71b, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.b67]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b67]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.458 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.458 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_42: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.dae = impl_witness_access constants.%Copy.impl_witness.f48, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_42, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc9(%.loc9_42)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc9: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
