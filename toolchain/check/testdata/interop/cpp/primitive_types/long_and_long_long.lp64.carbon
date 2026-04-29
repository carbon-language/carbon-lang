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
// CHECK:STDOUT:     .long = @F.%i64.2
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
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.95b = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.288]
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
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_50: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn.loc11_50 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:     .unsigned_long = @F.%u64.2
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
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.fec]
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
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_59: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn.loc11_59 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
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
// CHECK:STDOUT:   %As.impl_witness.158: <witness> = impl_witness imports.%As.impl_witness_table.eb3 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b3e = facet_value Core.IntLiteral, (%As.impl_witness.158) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.1b8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.c9c: type = fn_type_with_self_type %As.WithSelf.Convert.type.1b8, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.737: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.64f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.084: %ImplicitAs.type.139 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.737) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.566: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.084) [concrete]
// CHECK:STDOUT:   %.0a4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.566, %ImplicitAs.facet.084 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.cdc: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.eb3 = impl_witness_table (%Core.import_ref.cdc), @Core.IntLiteral.as.As.impl.823 [concrete]
// CHECK:STDOUT:   %Core.import_ref.74a: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.64f = impl_witness_table (%Core.import_ref.74a), @Cpp.long_long.as.ImplicitAs.impl.60b [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
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
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.c9c = impl_witness_access constants.%As.impl_witness.158, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.0a4 = impl_witness_access constants.%ImplicitAs.impl_witness.737, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6d4: <witness> = impl_witness imports.%Copy.impl_witness_table.804 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.6d4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.fdc: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3c6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.fdc, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7a9: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.804 = impl_witness_table (%Core.import_ref.7a9), @Cpp.long_long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.3c6 = impl_witness_access constants.%Copy.impl_witness.6d4, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.625: <witness> = impl_witness imports.%EqWith.impl_witness_table.636 [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.4be = facet_value %Cpp.long_long, (%EqWith.impl_witness.625) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.36e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.dac: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.36e, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.7de: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.49b: %Cpp.long_long.as.EqWith.impl.Equal.type.7de = struct_value () [concrete]
// CHECK:STDOUT:   %.df4: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.dac, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.4e3: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.608: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4e3 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.dab: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.366 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.65e = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.dab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.e0c: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.053: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.628: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.b0a: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %.7d6: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.628, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.445: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.d29: %Cpp.long_long.as.OrderedWith.impl.Greater.type.445 = struct_value () [concrete]
// CHECK:STDOUT:   %.5c4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.e0c, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a02: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.304: %Cpp.long_long.as.OrderedWith.impl.Less.type.a02 = struct_value () [concrete]
// CHECK:STDOUT:   %.7cc: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.b0a, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.cd4: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.cd4 = struct_value () [concrete]
// CHECK:STDOUT:   %.3fa: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.053, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e6d: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.de3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e6d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.35f: %Cpp.long_long.as.EqWith.impl.Equal.type.7de = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.49b]
// CHECK:STDOUT:   %Core.import_ref.1ba: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4e3 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.608]
// CHECK:STDOUT:   %EqWith.impl_witness_table.636 = impl_witness_table (%Core.import_ref.35f, %Core.import_ref.1ba), @Cpp.long_long.as.EqWith.impl.fb8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.064: %Cpp.long_long.as.OrderedWith.impl.Less.type.a02 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.304]
// CHECK:STDOUT:   %Core.import_ref.37c: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e6d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.de3]
// CHECK:STDOUT:   %Core.import_ref.cea: %Cpp.long_long.as.OrderedWith.impl.Greater.type.445 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.d29]
// CHECK:STDOUT:   %Core.import_ref.e7b: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.cd4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.366 = impl_witness_table (%Core.import_ref.064, %Core.import_ref.37c, %Core.import_ref.cea, %Core.import_ref.e7b), @Cpp.long_long.as.OrderedWith.impl.fd8 [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.604 = impl_witness_access constants.%EqWith.impl_witness.625, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.49b]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.df4 = impl_witness_access constants.%EqWith.impl_witness.625, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.608]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.7d6 = impl_witness_access constants.%OrderedWith.impl_witness.dab, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.d29]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.5c4 = impl_witness_access constants.%OrderedWith.impl_witness.dab, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.304]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.7cc = impl_witness_access constants.%OrderedWith.impl_witness.dab, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.3fa = impl_witness_access constants.%OrderedWith.impl_witness.dab, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.de3]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d68: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.ea5: %ImplicitAs.type.a03 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.67e(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bab317.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.67e(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bab317.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.67e(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.214d79.1: %Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.67e(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.214d79.2: %Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.ff9: <witness> = impl_witness imports.%EqWith.impl_witness_table.e2e, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.502737.1: %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.502737.2: %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.2e5: %EqWith.type.d68 = facet_value %Cpp.long_long, (%EqWith.impl_witness.ff9) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.60f: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.2e5) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.3c0: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.2e5) [concrete]
// CHECK:STDOUT:   %.ce4: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.60f, %EqWith.facet.2e5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.634a1c.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.502737.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.634a1c.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.502737.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.d6c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.3c0, %EqWith.facet.2e5 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.befbae.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.befbae.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.e34: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.56f4c1.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.56f4c1.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.6d5ff8.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.6d5ff8.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d90ce2.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d90ce2.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.95a450.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.79a(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.95a450.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.67f: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.71f, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0c8072.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0c8072.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.97d: %OrderedWith.type.e34 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.67f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.904: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.97d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.df8: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.97d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.546: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.97d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7ca: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.97d) [concrete]
// CHECK:STDOUT:   %.254: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.546, %OrderedWith.facet.97d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.1ff976.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.1ff976.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.aed: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.904, %OrderedWith.facet.97d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.52d741.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.0c8072.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.52d741.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.0c8072.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.b31: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7ca, %OrderedWith.facet.97d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e28331.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e28331.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.a86: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.df8, %OrderedWith.facet.97d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.abe2fe.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.abe2fe.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.504: <witness> = impl_witness imports.%EqWith.impl_witness_table.e2e, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.1: %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.67e(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.179: %EqWith.type.ded = facet_value %Cpp.long_long, (%EqWith.impl_witness.504) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.5e2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.179) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.510: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.179) [concrete]
// CHECK:STDOUT:   %.639: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.5e2, %EqWith.facet.179 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.92aa2e.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.92aa2e.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.f02: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.510, %EqWith.facet.179 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d6554.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d6554.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.18f: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.71f, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.938bb5.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.938bb5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.79a(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.a4a: %OrderedWith.type.358 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.18f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.86b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.a4a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.5bf: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.a4a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ed9: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.a4a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.6cb: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.a4a) [concrete]
// CHECK:STDOUT:   %.840: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ed9, %OrderedWith.facet.a4a [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.7855f8.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.7855f8.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.14b: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.86b, %OrderedWith.facet.a4a [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.9934b4.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.938bb5.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.9934b4.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.938bb5.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.cc0: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.6cb, %OrderedWith.facet.a4a [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ed5e.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ed5e.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.038: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.5bf, %OrderedWith.facet.a4a [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4b3629.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4b3629.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.fb2: @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.214d79.1)]
// CHECK:STDOUT:   %Core.import_ref.ad7: @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.bab317.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.e2e = impl_witness_table (%Core.import_ref.fb2, %Core.import_ref.ad7), @Cpp.long_long.as.EqWith.impl.67e [concrete]
// CHECK:STDOUT:   %Core.NotEqual.6a9: @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.bcc133.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.bab317.2)]
// CHECK:STDOUT:   %Core.Equal.72f: @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.6201cf.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.67e.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.214d79.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b58: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.95a450.1)]
// CHECK:STDOUT:   %Core.import_ref.1f9: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d90ce2.1)]
// CHECK:STDOUT:   %Core.import_ref.8d7: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.6d5ff8.1)]
// CHECK:STDOUT:   %Core.import_ref.7a9: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.56f4c1.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.71f = impl_witness_table (%Core.import_ref.b58, %Core.import_ref.1f9, %Core.import_ref.8d7, %Core.import_ref.7a9), @Cpp.long_long.as.OrderedWith.impl.79a [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.56b: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c38dd8.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.56f4c1.2)]
// CHECK:STDOUT:   %Core.Greater.a90: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.673884.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.6d5ff8.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.33c: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.9656b7.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d90ce2.2)]
// CHECK:STDOUT:   %Core.Less.0ea: @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.acb80a.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.79a.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.95a450.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.ce4 = impl_witness_access constants.%EqWith.impl_witness.ff9, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.502737.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.634a1c.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.1 = specific_constant imports.%Core.Equal.72f, @Cpp.long_long.as.EqWith.impl.67e(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.502737.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.c56f73.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.502737.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.634a1c.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.d6c = impl_witness_access constants.%EqWith.impl_witness.ff9, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.befbae.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.1 = specific_constant imports.%Core.NotEqual.6a9, @Cpp.long_long.as.EqWith.impl.67e(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.64e27b.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.1e17d3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.befbae.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.254 = impl_witness_access constants.%OrderedWith.impl_witness.67f, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.1ff976.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.1 = specific_constant imports.%Core.Greater.a90, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.81b99c.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.578bc4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.1ff976.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.aed = impl_witness_access constants.%OrderedWith.impl_witness.67f, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.0c8072.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.52d741.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.1 = specific_constant imports.%Core.Less.0ea, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.0c8072.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.9ca40a.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.0c8072.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.52d741.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.b31 = impl_witness_access constants.%OrderedWith.impl_witness.67f, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e28331.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.1 = specific_constant imports.%Core.GreaterOrEquivalent.56b, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.ad81a1.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c417c3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e28331.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.a86 = impl_witness_access constants.%OrderedWith.impl_witness.67f, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.abe2fe.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.1 = specific_constant imports.%Core.LessOrEquivalent.33c, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.985924.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1defb1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long_long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long_long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.abe2fe.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.639 = impl_witness_access constants.%EqWith.impl_witness.504, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.92aa2e.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.1 = specific_constant imports.%Core.Equal.72f, @Cpp.long_long.as.EqWith.impl.67e(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.bb381e.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.cdd8e5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.92aa2e.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.f02 = impl_witness_access constants.%EqWith.impl_witness.504, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d6554.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.1 = specific_constant imports.%Core.NotEqual.6a9, @Cpp.long_long.as.EqWith.impl.67e(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.d3031f.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.b6bcc7.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d6554.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.840 = impl_witness_access constants.%OrderedWith.impl_witness.18f, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.7855f8.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.1 = specific_constant imports.%Core.Greater.a90, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.8c4562.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.3ab52f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.7855f8.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.14b = impl_witness_access constants.%OrderedWith.impl_witness.18f, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.938bb5.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.9934b4.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.1 = specific_constant imports.%Core.Less.0ea, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.938bb5.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.ccbc54.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.938bb5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.9934b4.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.cc0 = impl_witness_access constants.%OrderedWith.impl_witness.18f, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ed5e.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.1 = specific_constant imports.%Core.GreaterOrEquivalent.56b, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.9ac5a4.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.f21cb0.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long_long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long_long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ed5e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.038 = impl_witness_access constants.%OrderedWith.impl_witness.18f, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4b3629.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.1 = specific_constant imports.%Core.LessOrEquivalent.33c, @Cpp.long_long.as.OrderedWith.impl.79a(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.cb98c7.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.451cbd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long_long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long_long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4b3629.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.ea5: %ImplicitAs.type.a03 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a66908.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.53c(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.8d0b9b.1: %T.as_type.as.EqWith.impl.NotEqual.type.a66908.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a66908.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.53c(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.8d0b9b.2: %T.as_type.as.EqWith.impl.NotEqual.type.a66908.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6b9708.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.53c(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.40c405.1: %T.as_type.as.EqWith.impl.Equal.type.6b9708.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6b9708.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.53c(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.40c405.2: %T.as_type.as.EqWith.impl.Equal.type.6b9708.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.cec: <witness> = impl_witness imports.%EqWith.impl_witness_table.c76, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6609ee.1: %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.078723.1: %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6609ee.2: %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.078723.2: %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.97b: %EqWith.type.4be = facet_value %i64, (%EqWith.impl_witness.cec) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.f4e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.97b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a3f: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.97b) [concrete]
// CHECK:STDOUT:   %.5a4: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.f4e, %EqWith.facet.97b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.92a53c.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6609ee.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.92a53c.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6609ee.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.573: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a3f, %EqWith.facet.97b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.230234.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.078723.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.230234.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.078723.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9b70f1.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9b70f1.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.a3a747.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.0253af.1: %T.as_type.as.OrderedWith.impl.Greater.type.a3a747.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.a3a747.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.0253af.2: %T.as_type.as.OrderedWith.impl.Greater.type.a3a747.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6b78dc.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6b78dc.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.09e4bc.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.621894.1: %T.as_type.as.OrderedWith.impl.Less.type.09e4bc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.09e4bc.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.144(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.621894.2: %T.as_type.as.OrderedWith.impl.Less.type.09e4bc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.413: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.426, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.953748.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.7ceb53.1: %T.as_type.as.OrderedWith.impl.Less.type.953748.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6a026c.1: %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.953748.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.7ceb53.2: %T.as_type.as.OrderedWith.impl.Less.type.953748.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6a026c.2: %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.35d: %OrderedWith.type.65e = facet_value %i64, (%OrderedWith.impl_witness.413) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ce0: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.35d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.1ea: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.35d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.aa0: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.35d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d9a: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.35d) [concrete]
// CHECK:STDOUT:   %.638: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.aa0, %OrderedWith.facet.35d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.e426c5.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.6a026c.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.e426c5.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.6a026c.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.171: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ce0, %OrderedWith.facet.35d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.26328c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.7ceb53.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.26328c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.7ceb53.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d9a, %OrderedWith.facet.35d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.cf5ee0.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.cf5ee0.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.e34: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.1ea, %OrderedWith.facet.35d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ee0adf.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ee0adf.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.360: <witness> = impl_witness imports.%EqWith.impl_witness_table.c76, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.414f7d.1: %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.4c54ab.1: %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.414f7d.2: %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.53c(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.4c54ab.2: %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.f7e: %EqWith.type.4be = facet_value Core.IntLiteral, (%EqWith.impl_witness.360) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.aba: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.f7e) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c66: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.f7e) [concrete]
// CHECK:STDOUT:   %.deb: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.aba, %EqWith.facet.f7e [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.38ba67.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.414f7d.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.414f7d.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.965a6d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.38ba67.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.414f7d.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.414f7d.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.965a6d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.2 [concrete]
// CHECK:STDOUT:   %.5ff: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c66, %EqWith.facet.f7e [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.b78a35.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.4c54ab.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.4c54ab.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.525f58.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.b78a35.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.4c54ab.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.4c54ab.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.525f58.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.a2b: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.426, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.23fbda.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.5e9a29.1: %T.as_type.as.OrderedWith.impl.Less.type.23fbda.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.965505.1: %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.23fbda.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.5e9a29.2: %T.as_type.as.OrderedWith.impl.Less.type.23fbda.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.965505.2: %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.144(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.04b: %OrderedWith.type.65e = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.a2b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c57: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.04b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.d75: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.04b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.2b9: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.04b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.50d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.04b) [concrete]
// CHECK:STDOUT:   %.0a3: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.2b9, %OrderedWith.facet.04b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.d28062.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.965505.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.965505.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.5f9ea0.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.d28062.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.965505.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.965505.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.5f9ea0.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.2 [concrete]
// CHECK:STDOUT:   %.add: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c57, %OrderedWith.facet.04b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.0457c8.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.5e9a29.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.5e9a29.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.c7f934.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.0457c8.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.5e9a29.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.5e9a29.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.c7f934.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.2 [concrete]
// CHECK:STDOUT:   %.679: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.50d, %OrderedWith.facet.04b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4e52dc.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.0a59bb.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4e52dc.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.0a59bb.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.2 [concrete]
// CHECK:STDOUT:   %.adc: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.d75, %OrderedWith.facet.04b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.990ba9.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.6f26dc.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.990ba9.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.6f26dc.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7de: @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.6b9708.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.40c405.1)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.a66908.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.8d0b9b.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.c76 = impl_witness_table (%Core.import_ref.7de, %Core.import_ref.b0a), @T.as_type.as.EqWith.impl.53c [concrete]
// CHECK:STDOUT:   %Core.NotEqual.728: @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.a66908.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.8d0b9b.2)]
// CHECK:STDOUT:   %Core.Equal.dca: @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.6b9708.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.53c.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.40c405.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.fa6e: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.09e4bc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.621894.1)]
// CHECK:STDOUT:   %Core.import_ref.1b07: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6b78dc.1)]
// CHECK:STDOUT:   %Core.import_ref.948: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.a3a747.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.0253af.1)]
// CHECK:STDOUT:   %Core.import_ref.ba48: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9b70f1.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.426 = impl_witness_table (%Core.import_ref.fa6e, %Core.import_ref.1b07, %Core.import_ref.948, %Core.import_ref.ba48), @T.as_type.as.OrderedWith.impl.144 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.4ff: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2ccf.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9b70f1.2)]
// CHECK:STDOUT:   %Core.Greater.5ce: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.a3a747.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.0253af.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.d58: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.6b5779.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6b78dc.2)]
// CHECK:STDOUT:   %Core.Less.030: @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.09e4bc.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.144.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.621894.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.5a4 = impl_witness_access constants.%EqWith.impl_witness.cec, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6609ee.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.92a53c.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.1 = specific_constant imports.%Core.Equal.dca, @T.as_type.as.EqWith.impl.53c(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6609ee.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.f1fa7b.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6609ee.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.92a53c.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.573 = impl_witness_access constants.%EqWith.impl_witness.cec, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.078723.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.230234.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.1 = specific_constant imports.%Core.NotEqual.728, @T.as_type.as.EqWith.impl.53c(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.078723.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.fc47d6.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.078723.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.230234.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.638 = impl_witness_access constants.%OrderedWith.impl_witness.413, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.6a026c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.e426c5.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.1 = specific_constant imports.%Core.Greater.5ce, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.6a026c.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.1b4ba9.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.6a026c.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.e426c5.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.171 = impl_witness_access constants.%OrderedWith.impl_witness.413, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7ceb53.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.26328c.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.953748.1 = specific_constant imports.%Core.Less.030, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7ceb53.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.953748.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7ceb53.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.26328c.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.205 = impl_witness_access constants.%OrderedWith.impl_witness.413, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.cf5ee0.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.1 = specific_constant imports.%Core.GreaterOrEquivalent.4ff, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.148554.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1e5e34.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.cf5ee0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.e34 = impl_witness_access constants.%OrderedWith.impl_witness.413, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ee0adf.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.1 = specific_constant imports.%Core.LessOrEquivalent.d58, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43e988.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fdb62b.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ee0adf.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.deb = impl_witness_access constants.%EqWith.impl_witness.360, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.414f7d.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.38ba67.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.965a6d.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.1 = specific_constant imports.%Core.Equal.dca, @T.as_type.as.EqWith.impl.53c(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.414f7d.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.f6c0c8.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.414f7d.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.38ba67.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.d032e7.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.965a6d.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.5ff = impl_witness_access constants.%EqWith.impl_witness.360, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.4c54ab.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.b78a35.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.525f58.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.1 = specific_constant imports.%Core.NotEqual.728, @T.as_type.as.EqWith.impl.53c(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.4c54ab.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.1072c0.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.4c54ab.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.b78a35.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.9df0c2.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.525f58.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.0a3 = impl_witness_access constants.%OrderedWith.impl_witness.a2b, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.965505.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.d28062.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.5f9ea0.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.1 = specific_constant imports.%Core.Greater.5ce, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.965505.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.9a9030.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.965505.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.d28062.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4b11f0.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.5f9ea0.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.add = impl_witness_access constants.%OrderedWith.impl_witness.a2b, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5e9a29.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.0457c8.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.c7f934.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.23fbda.1 = specific_constant imports.%Core.Less.030, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5e9a29.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.23fbda.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.5e9a29.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.0457c8.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.e2692e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.c7f934.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.679 = impl_witness_access constants.%OrderedWith.impl_witness.a2b, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4e52dc.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.0a59bb.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.1 = specific_constant imports.%Core.GreaterOrEquivalent.4ff, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.dbab37.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.adab83.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4e52dc.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95aa5c.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.0a59bb.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.adc = impl_witness_access constants.%OrderedWith.impl_witness.a2b, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.990ba9.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.6f26dc.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.1 = specific_constant imports.%Core.LessOrEquivalent.d58, @T.as_type.as.OrderedWith.impl.144(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.be1fca.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.12e809.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.990ba9.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4bb198.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.6f26dc.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
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
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.e0a: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.3a8: %T.as_type.as.ImplicitAs.impl.Convert.type.e0a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.2ec: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.6d4: <witness> = impl_witness imports.%Copy.impl_witness_table.804 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.6d4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f9a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fe1, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.86e: %ImplicitAs.type.a03 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.f9a) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.441: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.335, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.2ec = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.441) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.985: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9fe: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.985, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c22ab3.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c22ab3.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b35: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.f02835.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.f02835.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.5d6: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.eb8, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b35 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.5d6) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.726: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.4d2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.726, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2d753d.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2d753d.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.112: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.bba: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.b29, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.443299.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.443299.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.112 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.bba) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.ace: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.5fe: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.ace, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.c2e300.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.443299.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.c2e300.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.443299.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.45d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.498a51.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.498a51.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.ac9: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.f29, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.86595e.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.86595e.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.45d = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.ac9) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.6fe: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.592: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.6fe, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.9ad75a.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.86595e.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.9ad75a.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.86595e.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.a2e: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.832: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.969, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.129384.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.129384.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.a2e = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.832) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.706: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.56f: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.706, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.884783.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.129384.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.884783.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.129384.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.9bc: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.9b2: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.8e2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.9bc = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.9b2) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.2d5: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.712: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.2d5, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.e777fa.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.e777fa.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.219: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.cf2: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.97b, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.219 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.cf2) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.1ce: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.7f8: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.1ce, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.3d166b.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.3d166b.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.f9d: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.568: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.671, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.f9d = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.568) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.0f7: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.b7b: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.0f7, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.6de5bb.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.6de5bb.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.f24: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.d6a: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.dde, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.f24 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.d6a) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.ec1: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3f2: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.ec1, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f783c6.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f783c6.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.42e: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.58d: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.a01, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.42e = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.58d) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.34e: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9c4: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.34e, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.aa24ec.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.86e) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.aa24ec.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.86e) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.2a3: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.e0a) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.3a8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fe1 = impl_witness_table (%Core.import_ref.2a3), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.046: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.335 = impl_witness_table (%Core.import_ref.046), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.099: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2)]
// CHECK:STDOUT:   %Core.import_ref.7a9: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.804 = impl_witness_table (%Core.import_ref.7a9), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.849: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.f02835.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.eb8 = impl_witness_table (%Core.import_ref.849), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.938: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.f02835.2)]
// CHECK:STDOUT:   %Core.import_ref.cb1: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.b29 = impl_witness_table (%Core.import_ref.cb1), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5f4: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.2)]
// CHECK:STDOUT:   %Core.import_ref.add: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.498a51.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.f29 = impl_witness_table (%Core.import_ref.add), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.996: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.498a51.2)]
// CHECK:STDOUT:   %Core.import_ref.2a6: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.969 = impl_witness_table (%Core.import_ref.2a6), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.701: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.2)]
// CHECK:STDOUT:   %Core.import_ref.248: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.8e2 = impl_witness_table (%Core.import_ref.248), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a90: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.2)]
// CHECK:STDOUT:   %Core.import_ref.268: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.97b = impl_witness_table (%Core.import_ref.268), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.87d: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.2)]
// CHECK:STDOUT:   %Core.import_ref.a87: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.671 = impl_witness_table (%Core.import_ref.a87), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d1b: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.2)]
// CHECK:STDOUT:   %Core.import_ref.cbd: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.dde = impl_witness_table (%Core.import_ref.cbd), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d34: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.2)]
// CHECK:STDOUT:   %Core.import_ref.a05: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.a01 = impl_witness_table (%Core.import_ref.a05), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.807: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.9fe = impl_witness_access constants.%AddAssignWith.impl_witness.441, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c22ab3.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.1 = specific_constant imports.%Core.Op.099, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.fbd4aa.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.20a22f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.c22ab3.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.4d2 = impl_witness_access constants.%SubAssignWith.impl_witness.5d6, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2d753d.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.1 = specific_constant imports.%Core.Op.938, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.09a8e3.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.bdcde6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.2d753d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.5fe = impl_witness_access constants.%MulAssignWith.impl_witness.bba, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.443299.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.c2e300.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.1 = specific_constant imports.%Core.Op.5f4, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.443299.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e2c735.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.443299.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.c2e300.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.592 = impl_witness_access constants.%DivAssignWith.impl_witness.ac9, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.86595e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.9ad75a.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.1 = specific_constant imports.%Core.Op.996, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.86595e.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.376852.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.86595e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.9ad75a.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.56f = impl_witness_access constants.%ModAssignWith.impl_witness.832, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.129384.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.884783.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.1 = specific_constant imports.%Core.Op.701, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.129384.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.94b4c3.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.129384.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.884783.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.712 = impl_witness_access constants.%BitAndAssignWith.impl_witness.9b2, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.e777fa.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.1 = specific_constant imports.%Core.Op.a90, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cbbd71.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.26fab0.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.e777fa.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.7f8 = impl_witness_access constants.%BitOrAssignWith.impl_witness.cf2, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.3d166b.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.1 = specific_constant imports.%Core.Op.87d, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.436f6c.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.64f3c5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.3d166b.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.b7b = impl_witness_access constants.%BitXorAssignWith.impl_witness.568, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.6de5bb.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.1 = specific_constant imports.%Core.Op.d1b, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1f39ab.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.f0e9d1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.6de5bb.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.3f2 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.d6a, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f783c6.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.1 = specific_constant imports.%Core.Op.d34, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.369d19.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.98e8c0.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f783c6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.9c4 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.58d, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.f9a) [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.86e]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.aa24ec.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.1 = specific_constant imports.%Core.Op.807, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0c730c.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9852f6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.86e) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.aa24ec.2]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d84: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c30: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.335, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.c30) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.81a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.22f: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.81a, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.d66: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.f02835.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.f02835.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.98b: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.eb8, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.d66 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.98b) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.5d3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.dd8: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.5d3, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6a9965.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6a9965.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.e40: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.aee: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.b29, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.e40 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.aee) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.6aa: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.475: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.6aa, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.08c89b.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.08c89b.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.511: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.498a51.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.498a51.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.654: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.f29, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.511 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.654) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.173: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9b7: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.173, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.437985.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.437985.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2c3: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.884: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.969, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.85392b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.85392b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2c3 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.884) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ff8: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3ae: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ff8, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2e5057.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.85392b.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2e5057.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.85392b.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.8af: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.e50: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.8e2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.8af = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.e50) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.194: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.5d7: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.194, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.8d6e11.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.8d6e11.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.172: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.878: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.97b, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.172 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.878) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.639: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.744: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.639, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80f569.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80f569.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.2c8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.de9: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.671, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.2c8 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.de9) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.443: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e0d: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.443, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.e7b11a.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.e7b11a.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.b97: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.b1a: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.dde, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.b97 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.b1a) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.fe5: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.405: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.fe5, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1ffb17.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1ffb17.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.721: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.2b1: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.a01, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.721 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.2b1) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.0b1: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.84b: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.0b1, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.8ae17e.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.8ae17e.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.046: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.335 = impl_witness_table (%Core.import_ref.046), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.099: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.849: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.f02835.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.eb8 = impl_witness_table (%Core.import_ref.849), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.938: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.6d927b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.f02835.2)]
// CHECK:STDOUT:   %Core.import_ref.cb1: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.b29 = impl_witness_table (%Core.import_ref.cb1), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5f4: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.a601c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f8dfd6.2)]
// CHECK:STDOUT:   %Core.import_ref.add: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.498a51.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.f29 = impl_witness_table (%Core.import_ref.add), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.996: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.90bb4a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.498a51.2)]
// CHECK:STDOUT:   %Core.import_ref.2a6: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.969 = impl_witness_table (%Core.import_ref.2a6), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.701: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.1135b9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.b66b82.2)]
// CHECK:STDOUT:   %Core.import_ref.248: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.8e2 = impl_witness_table (%Core.import_ref.248), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a90: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a05c22.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.624b93.2)]
// CHECK:STDOUT:   %Core.import_ref.268: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.97b = impl_witness_table (%Core.import_ref.268), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.87d: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1d5518.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.cefc26.2)]
// CHECK:STDOUT:   %Core.import_ref.a87: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.671 = impl_witness_table (%Core.import_ref.a87), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d1b: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.8da076.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.53f9c8.2)]
// CHECK:STDOUT:   %Core.import_ref.cbd: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.dde = impl_witness_table (%Core.import_ref.cbd), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d34: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.fb5a0d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.9e2127.2)]
// CHECK:STDOUT:   %Core.import_ref.a05: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.a01 = impl_witness_table (%Core.import_ref.a05), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.807: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.40ffed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5d152f.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.22f = impl_witness_access constants.%AddAssignWith.impl_witness.c30, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = specific_constant imports.%Core.Op.099, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.dd8 = impl_witness_access constants.%SubAssignWith.impl_witness.98b, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6a9965.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.1 = specific_constant imports.%Core.Op.938, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.15b1da.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.e2f3a4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6a9965.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.475 = impl_witness_access constants.%MulAssignWith.impl_witness.aee, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.08c89b.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.1 = specific_constant imports.%Core.Op.5f4, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.fe73f6.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.f055d3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.08c89b.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.9b7 = impl_witness_access constants.%DivAssignWith.impl_witness.654, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.437985.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.1 = specific_constant imports.%Core.Op.996, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d85846.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.fd9eff.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.437985.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.3ae = impl_witness_access constants.%ModAssignWith.impl_witness.884, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.85392b.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2e5057.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.1 = specific_constant imports.%Core.Op.701, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.85392b.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.fde376.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.85392b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2e5057.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.5d7 = impl_witness_access constants.%BitAndAssignWith.impl_witness.e50, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.8d6e11.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.1 = specific_constant imports.%Core.Op.a90, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.8e2bf2.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.ea8d6a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long_long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long_long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.8d6e11.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.744 = impl_witness_access constants.%BitOrAssignWith.impl_witness.878, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80f569.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.1 = specific_constant imports.%Core.Op.87d, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.6483c2.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.ba1336.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80f569.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.e0d = impl_witness_access constants.%BitXorAssignWith.impl_witness.de9, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.e7b11a.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.1 = specific_constant imports.%Core.Op.d1b, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.fe202e.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.cdbb03.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.e7b11a.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.405 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.b1a, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1ffb17.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.1 = specific_constant imports.%Core.Op.d34, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.0fc301.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.6b432c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1ffb17.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.84b = impl_witness_access constants.%RightShiftAssignWith.impl_witness.2b1, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.8ae17e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.1 = specific_constant imports.%Core.Op.807, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.399089.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.5551b7.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.8ae17e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3c1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.335, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.3c1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.5d9: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.57c: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.5d9, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.1cd31a.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.1cd31a.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.046: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.335 = impl_witness_table (%Core.import_ref.046), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.099: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.57c = impl_witness_access constants.%AddAssignWith.impl_witness.3c1, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.1cd31a.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.1 = specific_constant imports.%Core.Op.099, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f8073b.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f8c91.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long_long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long_long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.1cd31a.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.a119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4af: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.d48) [concrete]
// CHECK:STDOUT:   %.a64: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4af, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d84: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c30: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.335, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.c30) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.81a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.22f: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.81a, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.046: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.335 = impl_witness_table (%Core.import_ref.046), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.099: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.25d3fe.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6b89af.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.22f = impl_witness_access constants.%AddAssignWith.impl_witness.c30, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = specific_constant imports.%Core.Op.099, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.16fdc3.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.a8260d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.8a57c0.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dcd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dcd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.1f7: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.11e: type = fn_type_with_self_type %Inc.WithSelf.Op.type.1f7, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.241: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.c6e: type = fn_type_with_self_type %Dec.WithSelf.Op.type.241, %Dec.facet [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.af8: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.af8), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.335: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.335), @Cpp.long_long.as.Dec.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.11e = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.c6e = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.438: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.195: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.438) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3d9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.195) [concrete]
// CHECK:STDOUT:   %.30f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3d9, %ImplicitAs.facet.195 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bcb: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f87: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bcb = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f87 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.c85(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl.c85(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.c85(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.042: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfe: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bb3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.408: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.dfe) [concrete]
// CHECK:STDOUT:   %.f17: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.408, %ImplicitAs.facet.dfe [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.141: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.042 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.042, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %As.impl_witness.d1f: <witness> = impl_witness imports.%As.impl_witness_table.002 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.45a = facet_value Core.IntLiteral, (%As.impl_witness.d1f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.8ce: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.83e: type = fn_type_with_self_type %As.WithSelf.Convert.type.8ce, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7c3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d12 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.29a: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.7c3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.314: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.29a) [concrete]
// CHECK:STDOUT:   %.66f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.314, %ImplicitAs.facet.29a [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7bd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bcb = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f87]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.517 = impl_witness_table (%Core.import_ref.7bd), @Core.IntLiteral.as.ImplicitAs.impl.45d [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.f0c073.1: %PassULongLong__carbon_thunk.type.88c6c2.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.887181.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl.c85 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.f0c073.2: %PassULongLong__carbon_thunk.type.88c6c2.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.887181.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.7e9: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.002 = impl_witness_table (%Core.import_ref.7e9), @Core.IntLiteral.as.As.impl.b6f [concrete]
// CHECK:STDOUT:   %Core.import_ref.5ac: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d12 = impl_witness_table (%Core.import_ref.5ac), @Cpp.unsigned_long_long.as.ImplicitAs.impl.fca [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.30f = impl_witness_access constants.%ImplicitAs.impl_witness.438, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f87]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b8]
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
// CHECK:STDOUT:   %impl.elem0.loc21: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.141]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.fec]
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
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc24_64.1: <bound method> = bound_method %float, %impl.elem0.loc24_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_64.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.83e = impl_witness_access constants.%As.impl_witness.d1f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.66f = impl_witness_access constants.%ImplicitAs.impl_witness.7c3, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.438: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.438) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3d9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.30f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3d9, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.79a: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.95c: <witness> = impl_witness imports.%Copy.impl_witness_table.759 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.95c) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.04c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.d24: type = fn_type_with_self_type %Copy.WithSelf.Op.type.04c, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7bd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.517 = impl_witness_table (%Core.import_ref.7bd), @Core.IntLiteral.as.ImplicitAs.impl.45d [concrete]
// CHECK:STDOUT:   %Core.import_ref.ffa: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.759 = impl_witness_table (%Core.import_ref.ffa), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.30f = impl_witness_access constants.%ImplicitAs.impl_witness.438, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.d24 = impl_witness_access constants.%Copy.impl_witness.95c, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
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
