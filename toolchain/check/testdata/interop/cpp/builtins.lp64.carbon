// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/builtins.lp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/builtins.lp64.carbon

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
  // CHECK:STDERR:       ^~~~~~~~~~
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
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc22_20: type = splice_block %long.ref.loc22 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc22: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc22: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc22_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc22_28.2: %i64 = converted %int_1.loc22, %.loc22_28.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long: %i64 = value_binding cpp_long, %.loc22_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.95b = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc23: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc25_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc25_50: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc25_50.1: <bound method> = bound_method %float, %impl.elem0.loc25_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc25_50: <specific function> = specific_function %impl.elem0.loc25_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_50.2: <bound method> = bound_method %float, %specific_fn.loc25_50 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc25_50.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25_50.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc25_50.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc25_50.4: init %f32.97e to %.loc25_50.3 = in_place_init %.loc25_50.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25_50.5: init %array_type to %a.var = array_init (%.loc25_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc25_3: init %array_type = converted %.loc25_50.1, %.loc25_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc25_3
// CHECK:STDOUT:   %.loc25_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc25: type = name_ref long, %i64.2 [concrete = constants.%i64]
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
// CHECK:STDOUT:   %int_1.loc37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc37_29: type = splice_block %unsigned_long.ref.loc37 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc37: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc37: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc37_46.1: <bound method> = bound_method %int_1.loc37, %impl.elem0.loc37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_46.2: <bound method> = bound_method %int_1.loc37, %specific_fn.loc37 [concrete = constants.%bound_method.fec]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc37_46.2(%int_1.loc37) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc37_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc37_46.2: %u64 = converted %int_1.loc37, %.loc37_46.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = value_binding cpp_unsigned_long, %.loc37_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc38: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc40_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc40_59: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc40_59.1: <bound method> = bound_method %float, %impl.elem0.loc40_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc40_59: <specific function> = specific_function %impl.elem0.loc40_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_59.2: <bound method> = bound_method %float, %specific_fn.loc40_59 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc40_59.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc40_59.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc40_59.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc40_59.4: init %f32.97e to %.loc40_59.3 = in_place_init %.loc40_59.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc40_59.5: init %array_type to %a.var = array_init (%.loc40_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc40_3: init %array_type = converted %.loc40_59.1, %.loc40_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc40_3
// CHECK:STDOUT:   %.loc40_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc40: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc40: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
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
// CHECK:STDOUT:   %int_1.loc59: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc59_25: type = splice_block %long_long.ref.loc59 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc59: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc59: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc59: <bound method> = bound_method %int_1.loc59, %impl.elem0.loc59 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc59(%int_1.loc59) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc59_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc59_38.2: %Cpp.long_long = converted %int_1.loc59, %.loc59_38.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = value_binding cpp_long_long, %.loc59_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.257 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc60_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc60: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc60: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc60_87.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc60: %ptr.a95 = addr_of %.loc60_87.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc60: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_long_long.ref.loc60, %addr.loc60)
// CHECK:STDOUT:   %.loc60_87.2: init %LongLongResult to %.loc60_87.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc60
// CHECK:STDOUT:   %.loc60_39: type = splice_block %LongLongResult.ref.loc60 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc60_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc60: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc60_87.3: ref %LongLongResult = temporary %.loc60_87.1, %.loc60_87.2
// CHECK:STDOUT:   %.loc60_87.4: %LongLongResult = acquire_value %.loc60_87.3
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = value_binding cpp_long_long_result, %.loc60_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long.patt: %pattern_type.76e = value_binding_pattern cpp_compat_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc62: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc62: type = splice_block %LongLong64.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %LongLong64.ref: type = name_ref LongLong64, imports.%Core.LongLong64 [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long: %Cpp.long_long = value_binding cpp_compat_long_long, %cpp_long_long.ref.loc62
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.257 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc63_64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc63: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc63_101.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc63: %ptr.a95 = addr_of %.loc63_101.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc63: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_compat_long_long.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_101.2: init %LongLongResult to %.loc63_101.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc63
// CHECK:STDOUT:   %.loc63_46: type = splice_block %LongLongResult.ref.loc63 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc63_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc63: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63_101.3: ref %LongLongResult = temporary %.loc63_101.1, %.loc63_101.2
// CHECK:STDOUT:   %.loc63_101.4: %LongLongResult = acquire_value %.loc63_101.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = value_binding cpp_compat_long_long_result, %.loc63_101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc65: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc65_42: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc65_39.1: %i64 = as_compatible %cpp_long_long.ref.loc65
// CHECK:STDOUT:   %.loc65_39.2: %i64 = converted %cpp_long_long.ref.loc65, %.loc65_39.1
// CHECK:STDOUT:   %i64.loc65_19: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc65_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.cd9 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc66_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc66: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc66_77.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc66: %ptr.305 = addr_of %.loc66_77.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc66: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.2(%carbon_i64.ref, %addr.loc66)
// CHECK:STDOUT:   %.loc66_77.2: init %LongResult to %.loc66_77.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc66
// CHECK:STDOUT:   %.loc66_36: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc66_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc66_77.3: ref %LongResult = temporary %.loc66_77.1, %.loc66_77.2
// CHECK:STDOUT:   %.loc66_77.4: %LongResult = acquire_value %.loc66_77.3
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = value_binding carbon_i64_result, %.loc66_77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc68_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc68_55: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc68_55.1: <bound method> = bound_method %float, %impl.elem0.loc68_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc68_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_55.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc68_55.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc68_55.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc68_55.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc68_55.4: init %f32.97e to %.loc68_55.3 = in_place_init %.loc68_55.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc68_55.5: init %array_type to %a.var = array_init (%.loc68_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc68_3: init %array_type = converted %.loc68_55.1, %.loc68_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc68_3
// CHECK:STDOUT:   %.loc68_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc68: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc68: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc68: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc68_30.1: %.c9c = impl_witness_access constants.%As.impl_witness.158, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc68_30.1: <bound method> = bound_method %int_1.loc68, %impl.elem0.loc68_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc68_30.1(%int_1.loc68) [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc68_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc68_30.2: %Cpp.long_long = converted %int_1.loc68, %.loc68_30.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %impl.elem0.loc68_30.2: %.0a4 = impl_witness_access constants.%ImplicitAs.impl_witness.737, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc68_30.2: <bound method> = bound_method %.loc68_30.2, %impl.elem0.loc68_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc68_30.2(%.loc68_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc68_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc68_30.4: Core.IntLiteral = converted %.loc68_30.2, %.loc68_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc68_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound: <bound method> = bound_method %.loc66_77.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call %LongResult.cpp_destructor.bound(%.loc66_77.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63_101.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc63: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc63(%.loc63_101.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc60: <bound method> = bound_method %.loc60_87.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc60: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc60(%.loc60_87.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc99: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc99: <bound method> = bound_method %int_1, %impl.elem0.loc99 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc99(%int_1) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc99_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc99_3
// CHECK:STDOUT:   %.loc99_13: type = splice_block %long_long.ref.loc99 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc99: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc99: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.76e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc100_33: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc100: %.3c6 = impl_witness_access constants.%Copy.impl_witness.6d4, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc100: <bound method> = bound_method %.loc100_33, %impl.elem0.loc100
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc100(%.loc100_33)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc100_20: type = splice_block %long_long.ref.loc100 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc100: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc100: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc100: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc100: init %empty_tuple.type = call %Destroy.Op.bound.loc100(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc99: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc99: init %empty_tuple.type = call %Destroy.Op.bound.loc99(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
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
// CHECK:STDOUT:   %int_1.loc112: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc112_13: type = splice_block %long_long.ref.loc112 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc112: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc112: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc112: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc112: <bound method> = bound_method %int_1.loc112, %impl.elem0.loc112 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc112: init %Cpp.long_long = call %bound_method.loc112(%int_1.loc112) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc112_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc112 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc112_26.2: %Cpp.long_long = converted %int_1.loc112, %.loc112_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc112_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc113: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc113_13: type = splice_block %long_long.ref.loc113 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc113: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc113: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc113: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc113: <bound method> = bound_method %int_1.loc113, %impl.elem0.loc113 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc113: init %Cpp.long_long = call %bound_method.loc113(%int_1.loc113) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc113_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc113 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc113_26.2: %Cpp.long_long = converted %int_1.loc113, %.loc113_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc113_26.2
// CHECK:STDOUT:   %a.ref.loc114: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc114: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc114: %.604 = impl_witness_access constants.%EqWith.impl_witness.625, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.49b]
// CHECK:STDOUT:   %bound_method.loc114: <bound method> = bound_method %a.ref.loc114, %impl.elem0.loc114
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc114(%a.ref.loc114, %b.ref.loc114)
// CHECK:STDOUT:   %a.ref.loc115: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc115: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc115: %.df4 = impl_witness_access constants.%EqWith.impl_witness.625, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.608]
// CHECK:STDOUT:   %bound_method.loc115: <bound method> = bound_method %a.ref.loc115, %impl.elem1.loc115
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc115(%a.ref.loc115, %b.ref.loc115)
// CHECK:STDOUT:   %a.ref.loc116: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc116: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.7d6 = impl_witness_access constants.%OrderedWith.impl_witness.dab, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.d29]
// CHECK:STDOUT:   %bound_method.loc116: <bound method> = bound_method %a.ref.loc116, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc116(%a.ref.loc116, %b.ref.loc116)
// CHECK:STDOUT:   %a.ref.loc117: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc117: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc117: %.5c4 = impl_witness_access constants.%OrderedWith.impl_witness.dab, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.304]
// CHECK:STDOUT:   %bound_method.loc117: <bound method> = bound_method %a.ref.loc117, %impl.elem0.loc117
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc117(%a.ref.loc117, %b.ref.loc117)
// CHECK:STDOUT:   %a.ref.loc118: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc118: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.7cc = impl_witness_access constants.%OrderedWith.impl_witness.dab, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc]
// CHECK:STDOUT:   %bound_method.loc118: <bound method> = bound_method %a.ref.loc118, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc118(%a.ref.loc118, %b.ref.loc118)
// CHECK:STDOUT:   %a.ref.loc119: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc119: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc119: %.3fa = impl_witness_access constants.%OrderedWith.impl_witness.dab, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.de3]
// CHECK:STDOUT:   %bound_method.loc119: <bound method> = bound_method %a.ref.loc119, %impl.elem1.loc119
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc119(%a.ref.loc119, %b.ref.loc119)
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
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.1bc(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.1bc(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.1bc(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.7b9c94.1: %Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.1bc(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.7b9c94.2: %Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.756: <witness> = impl_witness imports.%EqWith.impl_witness_table.902, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.610927.1: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.610927.2: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.d0f: %EqWith.type.d68 = facet_value %Cpp.long_long, (%EqWith.impl_witness.756) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.0d2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.d0f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0eb: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.d0f) [concrete]
// CHECK:STDOUT:   %.357: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.0d2, %EqWith.facet.d0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.610927.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.610927.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.d50: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0eb, %EqWith.facet.d0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.e34: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.c164d2.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%T.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.c164d2.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.940: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.a1b, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.39610d.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.136328.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.39610d.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.136328.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.149: %OrderedWith.type.e34 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.940) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.d11: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.149) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3e3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.149) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.4a1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.149) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.065: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.149) [concrete]
// CHECK:STDOUT:   %.e0d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.4a1, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.136328.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.136328.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.c1d: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.d11, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.39610d.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.39610d.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.f24: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.065, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.6b0: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3e3, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.5f3: <witness> = impl_witness imports.%EqWith.impl_witness_table.902, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.3190b1.1: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.3190b1.2: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.1bc(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.8cf: %EqWith.type.ded = facet_value %Cpp.long_long, (%EqWith.impl_witness.5f3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.fe8: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.8cf) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.8c1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.8cf) [concrete]
// CHECK:STDOUT:   %.b86: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.fe8, %EqWith.facet.8cf [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.3190b1.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.3190b1.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.251: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.8c1, %EqWith.facet.8cf [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.31a: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.a1b, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.ef6(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.8a0: %OrderedWith.type.358 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.31a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.a91: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.8a0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.6cb: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.8a0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.a55: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.8a0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.2c3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.8a0) [concrete]
// CHECK:STDOUT:   %.5d9: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.a55, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.885: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.a91, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.ea1: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.2c3, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.da4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.6cb, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.142: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.7b9c94.1)]
// CHECK:STDOUT:   %Core.import_ref.71f: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.902 = impl_witness_table (%Core.import_ref.142, %Core.import_ref.71f), @Cpp.long_long.as.EqWith.impl.1bc [concrete]
// CHECK:STDOUT:   %Core.NotEqual.dea: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.2)]
// CHECK:STDOUT:   %Core.Equal.357: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.7b9c94.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.81d: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.c164d2.1)]
// CHECK:STDOUT:   %Core.import_ref.21c: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.1)]
// CHECK:STDOUT:   %Core.import_ref.298: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.1)]
// CHECK:STDOUT:   %Core.import_ref.e3de: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.a1b = impl_witness_table (%Core.import_ref.81d, %Core.import_ref.21c, %Core.import_ref.298, %Core.import_ref.e3de), @Cpp.long_long.as.OrderedWith.impl.ef6 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.489: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.2)]
// CHECK:STDOUT:   %Core.Greater.a10: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.aaf: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.2)]
// CHECK:STDOUT:   %Core.Less.511: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.c164d2.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc132: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc132_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc132: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc132: <bound method> = bound_method %int_1.loc132, %impl.elem0.loc132 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc132: init %Cpp.long_long = call %bound_method.loc132(%int_1.loc132) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc132_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc132 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc132_26.2: %Cpp.long_long = converted %int_1.loc132, %.loc132_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc132_26.2
// CHECK:STDOUT:   %a.ref.loc133: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc133: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc133_5.1: %.357 = impl_witness_access constants.%EqWith.impl_witness.756, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.2]
// CHECK:STDOUT:   %bound_method.loc133_5.1: <bound method> = bound_method %a.ref.loc133, %impl.elem0.loc133_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc133: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc133_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc133 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc133: <specific function> = specific_function %impl.elem0.loc133_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.1]
// CHECK:STDOUT:   %bound_method.loc133_5.2: <bound method> = bound_method %a.ref.loc133, %specific_fn.loc133
// CHECK:STDOUT:   %.loc133_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = specific_constant imports.%Core.Equal.357, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Equal.ref.loc133: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = name_ref Equal, %.loc133_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc133: <bound method> = bound_method %a.ref.loc133, %Equal.ref.loc133
// CHECK:STDOUT:   %impl.elem0.loc133_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc133_5.3: <bound method> = bound_method %b.ref.loc133, %impl.elem0.loc133_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc133_5: init %Cpp.long_long = call %bound_method.loc133_5.3(%b.ref.loc133)
// CHECK:STDOUT:   %.loc133_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc133_5
// CHECK:STDOUT:   %.loc133_5.4: %Cpp.long_long = converted %b.ref.loc133, %.loc133_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc133: <specific function> = specific_function %Equal.ref.loc133, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.2]
// CHECK:STDOUT:   %bound_method.loc133_5.4: <bound method> = bound_method %a.ref.loc133, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc133
// CHECK:STDOUT:   %impl.elem0.loc133_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc133_8: <bound method> = bound_method %b.ref.loc133, %impl.elem0.loc133_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc133_8: init %Cpp.long_long = call %bound_method.loc133_8(%b.ref.loc133)
// CHECK:STDOUT:   %.loc133_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc133_8
// CHECK:STDOUT:   %.loc133_8.2: %Cpp.long_long = converted %b.ref.loc133, %.loc133_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc133: init bool = call %bound_method.loc133_5.4(%a.ref.loc133, %.loc133_8.2)
// CHECK:STDOUT:   %a.ref.loc134: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc134: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc134: %.d50 = impl_witness_access constants.%EqWith.impl_witness.756, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2]
// CHECK:STDOUT:   %bound_method.loc134_5.1: <bound method> = bound_method %a.ref.loc134, %impl.elem1.loc134
// CHECK:STDOUT:   %ImplicitAs.facet.loc134: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc134_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc134 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc134: <specific function> = specific_function %impl.elem1.loc134, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.1]
// CHECK:STDOUT:   %bound_method.loc134_5.2: <bound method> = bound_method %a.ref.loc134, %specific_fn.loc134
// CHECK:STDOUT:   %.loc134_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = specific_constant imports.%Core.NotEqual.dea, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %NotEqual.ref.loc134: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = name_ref NotEqual, %.loc134_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc134: <bound method> = bound_method %a.ref.loc134, %NotEqual.ref.loc134
// CHECK:STDOUT:   %impl.elem0.loc134_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc134_5.3: <bound method> = bound_method %b.ref.loc134, %impl.elem0.loc134_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc134_5: init %Cpp.long_long = call %bound_method.loc134_5.3(%b.ref.loc134)
// CHECK:STDOUT:   %.loc134_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc134_5
// CHECK:STDOUT:   %.loc134_5.4: %Cpp.long_long = converted %b.ref.loc134, %.loc134_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc134: <specific function> = specific_function %NotEqual.ref.loc134, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.2]
// CHECK:STDOUT:   %bound_method.loc134_5.4: <bound method> = bound_method %a.ref.loc134, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc134
// CHECK:STDOUT:   %impl.elem0.loc134_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc134_8: <bound method> = bound_method %b.ref.loc134, %impl.elem0.loc134_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc134_8: init %Cpp.long_long = call %bound_method.loc134_8(%b.ref.loc134)
// CHECK:STDOUT:   %.loc134_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc134_8
// CHECK:STDOUT:   %.loc134_8.2: %Cpp.long_long = converted %b.ref.loc134, %.loc134_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc134: init bool = call %bound_method.loc134_5.4(%a.ref.loc134, %.loc134_8.2)
// CHECK:STDOUT:   %a.ref.loc135: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc135: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc135: %.e0d = impl_witness_access constants.%OrderedWith.impl_witness.940, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.2]
// CHECK:STDOUT:   %bound_method.loc135_5.1: <bound method> = bound_method %a.ref.loc135, %impl.elem2.loc135
// CHECK:STDOUT:   %ImplicitAs.facet.loc135: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc135_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc135 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc135: <specific function> = specific_function %impl.elem2.loc135, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.1]
// CHECK:STDOUT:   %bound_method.loc135_5.2: <bound method> = bound_method %a.ref.loc135, %specific_fn.loc135
// CHECK:STDOUT:   %.loc135_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = specific_constant imports.%Core.Greater.a10, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Greater.ref.loc135: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = name_ref Greater, %.loc135_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc135: <bound method> = bound_method %a.ref.loc135, %Greater.ref.loc135
// CHECK:STDOUT:   %impl.elem0.loc135_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc135_5.3: <bound method> = bound_method %b.ref.loc135, %impl.elem0.loc135_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc135_5: init %Cpp.long_long = call %bound_method.loc135_5.3(%b.ref.loc135)
// CHECK:STDOUT:   %.loc135_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc135_5
// CHECK:STDOUT:   %.loc135_5.4: %Cpp.long_long = converted %b.ref.loc135, %.loc135_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc135: <specific function> = specific_function %Greater.ref.loc135, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.2]
// CHECK:STDOUT:   %bound_method.loc135_5.4: <bound method> = bound_method %a.ref.loc135, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc135
// CHECK:STDOUT:   %impl.elem0.loc135_7: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc135_7: <bound method> = bound_method %b.ref.loc135, %impl.elem0.loc135_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc135_7: init %Cpp.long_long = call %bound_method.loc135_7(%b.ref.loc135)
// CHECK:STDOUT:   %.loc135_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc135_7
// CHECK:STDOUT:   %.loc135_7.2: %Cpp.long_long = converted %b.ref.loc135, %.loc135_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc135: init bool = call %bound_method.loc135_5.4(%a.ref.loc135, %.loc135_7.2)
// CHECK:STDOUT:   %a.ref.loc136: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc136: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc136_5.1: %.c1d = impl_witness_access constants.%OrderedWith.impl_witness.940, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.2]
// CHECK:STDOUT:   %bound_method.loc136_5.1: <bound method> = bound_method %a.ref.loc136, %impl.elem0.loc136_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc136: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc136_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc136 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc136: <specific function> = specific_function %impl.elem0.loc136_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.1]
// CHECK:STDOUT:   %bound_method.loc136_5.2: <bound method> = bound_method %a.ref.loc136, %specific_fn.loc136
// CHECK:STDOUT:   %.loc136_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = specific_constant imports.%Core.Less.511, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Less.ref.loc136: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = name_ref Less, %.loc136_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc136: <bound method> = bound_method %a.ref.loc136, %Less.ref.loc136
// CHECK:STDOUT:   %impl.elem0.loc136_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc136_5.3: <bound method> = bound_method %b.ref.loc136, %impl.elem0.loc136_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc136_5: init %Cpp.long_long = call %bound_method.loc136_5.3(%b.ref.loc136)
// CHECK:STDOUT:   %.loc136_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc136_5
// CHECK:STDOUT:   %.loc136_5.4: %Cpp.long_long = converted %b.ref.loc136, %.loc136_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc136: <specific function> = specific_function %Less.ref.loc136, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.2]
// CHECK:STDOUT:   %bound_method.loc136_5.4: <bound method> = bound_method %a.ref.loc136, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc136
// CHECK:STDOUT:   %impl.elem0.loc136_7: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc136_7: <bound method> = bound_method %b.ref.loc136, %impl.elem0.loc136_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc136_7: init %Cpp.long_long = call %bound_method.loc136_7(%b.ref.loc136)
// CHECK:STDOUT:   %.loc136_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc136_7
// CHECK:STDOUT:   %.loc136_7.2: %Cpp.long_long = converted %b.ref.loc136, %.loc136_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc136: init bool = call %bound_method.loc136_5.4(%a.ref.loc136, %.loc136_7.2)
// CHECK:STDOUT:   %a.ref.loc137: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc137: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc137: %.f24 = impl_witness_access constants.%OrderedWith.impl_witness.940, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2]
// CHECK:STDOUT:   %bound_method.loc137_5.1: <bound method> = bound_method %a.ref.loc137, %impl.elem3.loc137
// CHECK:STDOUT:   %ImplicitAs.facet.loc137: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc137_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc137 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc137: <specific function> = specific_function %impl.elem3.loc137, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.1]
// CHECK:STDOUT:   %bound_method.loc137_5.2: <bound method> = bound_method %a.ref.loc137, %specific_fn.loc137
// CHECK:STDOUT:   %.loc137_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = specific_constant imports.%Core.GreaterOrEquivalent.489, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc137: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = name_ref GreaterOrEquivalent, %.loc137_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc137: <bound method> = bound_method %a.ref.loc137, %GreaterOrEquivalent.ref.loc137
// CHECK:STDOUT:   %impl.elem0.loc137_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc137_5.3: <bound method> = bound_method %b.ref.loc137, %impl.elem0.loc137_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc137_5: init %Cpp.long_long = call %bound_method.loc137_5.3(%b.ref.loc137)
// CHECK:STDOUT:   %.loc137_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc137_5
// CHECK:STDOUT:   %.loc137_5.4: %Cpp.long_long = converted %b.ref.loc137, %.loc137_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc137: <specific function> = specific_function %GreaterOrEquivalent.ref.loc137, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.2]
// CHECK:STDOUT:   %bound_method.loc137_5.4: <bound method> = bound_method %a.ref.loc137, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc137
// CHECK:STDOUT:   %impl.elem0.loc137_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc137_8: <bound method> = bound_method %b.ref.loc137, %impl.elem0.loc137_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc137_8: init %Cpp.long_long = call %bound_method.loc137_8(%b.ref.loc137)
// CHECK:STDOUT:   %.loc137_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc137_8
// CHECK:STDOUT:   %.loc137_8.2: %Cpp.long_long = converted %b.ref.loc137, %.loc137_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc137: init bool = call %bound_method.loc137_5.4(%a.ref.loc137, %.loc137_8.2)
// CHECK:STDOUT:   %a.ref.loc138: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc138: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc138: %.6b0 = impl_witness_access constants.%OrderedWith.impl_witness.940, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.2]
// CHECK:STDOUT:   %bound_method.loc138_5.1: <bound method> = bound_method %a.ref.loc138, %impl.elem1.loc138
// CHECK:STDOUT:   %ImplicitAs.facet.loc138: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc138_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc138 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc138: <specific function> = specific_function %impl.elem1.loc138, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.1]
// CHECK:STDOUT:   %bound_method.loc138_5.2: <bound method> = bound_method %a.ref.loc138, %specific_fn.loc138
// CHECK:STDOUT:   %.loc138_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = specific_constant imports.%Core.LessOrEquivalent.aaf, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc138: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = name_ref LessOrEquivalent, %.loc138_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc138: <bound method> = bound_method %a.ref.loc138, %LessOrEquivalent.ref.loc138
// CHECK:STDOUT:   %impl.elem0.loc138_5: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc138_5.3: <bound method> = bound_method %b.ref.loc138, %impl.elem0.loc138_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc138_5: init %Cpp.long_long = call %bound_method.loc138_5.3(%b.ref.loc138)
// CHECK:STDOUT:   %.loc138_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc138_5
// CHECK:STDOUT:   %.loc138_5.4: %Cpp.long_long = converted %b.ref.loc138, %.loc138_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc138: <specific function> = specific_function %LessOrEquivalent.ref.loc138, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.2]
// CHECK:STDOUT:   %bound_method.loc138_5.4: <bound method> = bound_method %a.ref.loc138, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc138
// CHECK:STDOUT:   %impl.elem0.loc138_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc138_8: <bound method> = bound_method %b.ref.loc138, %impl.elem0.loc138_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc138_8: init %Cpp.long_long = call %bound_method.loc138_8(%b.ref.loc138)
// CHECK:STDOUT:   %.loc138_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc138_8
// CHECK:STDOUT:   %.loc138_8.2: %Cpp.long_long = converted %b.ref.loc138, %.loc138_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc138: init bool = call %bound_method.loc138_5.4(%a.ref.loc138, %.loc138_8.2)
// CHECK:STDOUT:   %a.ref.loc140: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc140: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc140_5.1: %.b86 = impl_witness_access constants.%EqWith.impl_witness.5f3, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.2]
// CHECK:STDOUT:   %bound_method.loc140_5.1: <bound method> = bound_method %a.ref.loc140, %impl.elem0.loc140_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc140: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc140_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc140 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc140: <specific function> = specific_function %impl.elem0.loc140_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.1]
// CHECK:STDOUT:   %bound_method.loc140_5.2: <bound method> = bound_method %a.ref.loc140, %specific_fn.loc140
// CHECK:STDOUT:   %.loc140_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1 = specific_constant imports.%Core.Equal.357, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.1]
// CHECK:STDOUT:   %Equal.ref.loc140: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1 = name_ref Equal, %.loc140_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc140: <bound method> = bound_method %a.ref.loc140, %Equal.ref.loc140
// CHECK:STDOUT:   %impl.elem0.loc140_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc140_5.3: <bound method> = bound_method %int_1.loc140, %impl.elem0.loc140_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc140_5: init %Cpp.long_long = call %bound_method.loc140_5.3(%int_1.loc140) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc140_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc140_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc140_5.4: %Cpp.long_long = converted %int_1.loc140, %.loc140_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc140: <specific function> = specific_function %Equal.ref.loc140, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.2]
// CHECK:STDOUT:   %bound_method.loc140_5.4: <bound method> = bound_method %a.ref.loc140, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc140
// CHECK:STDOUT:   %impl.elem0.loc140_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc140_8: <bound method> = bound_method %int_1.loc140, %impl.elem0.loc140_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc140_8: init %Cpp.long_long = call %bound_method.loc140_8(%int_1.loc140) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc140_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc140_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc140_8.2: %Cpp.long_long = converted %int_1.loc140, %.loc140_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc140: init bool = call %bound_method.loc140_5.4(%a.ref.loc140, %.loc140_8.2)
// CHECK:STDOUT:   %a.ref.loc141: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc141: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc141: %.251 = impl_witness_access constants.%EqWith.impl_witness.5f3, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.2]
// CHECK:STDOUT:   %bound_method.loc141_5.1: <bound method> = bound_method %a.ref.loc141, %impl.elem1.loc141
// CHECK:STDOUT:   %ImplicitAs.facet.loc141: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc141_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc141 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc141: <specific function> = specific_function %impl.elem1.loc141, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.1]
// CHECK:STDOUT:   %bound_method.loc141_5.2: <bound method> = bound_method %a.ref.loc141, %specific_fn.loc141
// CHECK:STDOUT:   %.loc141_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1 = specific_constant imports.%Core.NotEqual.dea, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc141: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1 = name_ref NotEqual, %.loc141_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc141: <bound method> = bound_method %a.ref.loc141, %NotEqual.ref.loc141
// CHECK:STDOUT:   %impl.elem0.loc141_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc141_5.3: <bound method> = bound_method %int_1.loc141, %impl.elem0.loc141_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_5: init %Cpp.long_long = call %bound_method.loc141_5.3(%int_1.loc141) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc141_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc141_5.4: %Cpp.long_long = converted %int_1.loc141, %.loc141_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc141: <specific function> = specific_function %NotEqual.ref.loc141, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.2]
// CHECK:STDOUT:   %bound_method.loc141_5.4: <bound method> = bound_method %a.ref.loc141, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc141
// CHECK:STDOUT:   %impl.elem0.loc141_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc141_8: <bound method> = bound_method %int_1.loc141, %impl.elem0.loc141_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_8: init %Cpp.long_long = call %bound_method.loc141_8(%int_1.loc141) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc141_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc141_8.2: %Cpp.long_long = converted %int_1.loc141, %.loc141_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc141: init bool = call %bound_method.loc141_5.4(%a.ref.loc141, %.loc141_8.2)
// CHECK:STDOUT:   %a.ref.loc142: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc142: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc142: %.5d9 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.2]
// CHECK:STDOUT:   %bound_method.loc142_5.1: <bound method> = bound_method %a.ref.loc142, %impl.elem2.loc142
// CHECK:STDOUT:   %ImplicitAs.facet.loc142: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc142_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc142 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc142: <specific function> = specific_function %impl.elem2.loc142, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.1]
// CHECK:STDOUT:   %bound_method.loc142_5.2: <bound method> = bound_method %a.ref.loc142, %specific_fn.loc142
// CHECK:STDOUT:   %.loc142_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1 = specific_constant imports.%Core.Greater.a10, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1]
// CHECK:STDOUT:   %Greater.ref.loc142: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1 = name_ref Greater, %.loc142_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc142: <bound method> = bound_method %a.ref.loc142, %Greater.ref.loc142
// CHECK:STDOUT:   %impl.elem0.loc142_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc142_5.3: <bound method> = bound_method %int_1.loc142, %impl.elem0.loc142_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_5: init %Cpp.long_long = call %bound_method.loc142_5.3(%int_1.loc142) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc142_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc142_5.4: %Cpp.long_long = converted %int_1.loc142, %.loc142_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc142: <specific function> = specific_function %Greater.ref.loc142, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.2]
// CHECK:STDOUT:   %bound_method.loc142_5.4: <bound method> = bound_method %a.ref.loc142, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc142
// CHECK:STDOUT:   %impl.elem0.loc142_7: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc142_7: <bound method> = bound_method %int_1.loc142, %impl.elem0.loc142_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_7: init %Cpp.long_long = call %bound_method.loc142_7(%int_1.loc142) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc142_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc142_7.2: %Cpp.long_long = converted %int_1.loc142, %.loc142_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc142: init bool = call %bound_method.loc142_5.4(%a.ref.loc142, %.loc142_7.2)
// CHECK:STDOUT:   %a.ref.loc143: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc143: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc143_5.1: %.885 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.2]
// CHECK:STDOUT:   %bound_method.loc143_5.1: <bound method> = bound_method %a.ref.loc143, %impl.elem0.loc143_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc143: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc143_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc143 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc143: <specific function> = specific_function %impl.elem0.loc143_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.1]
// CHECK:STDOUT:   %bound_method.loc143_5.2: <bound method> = bound_method %a.ref.loc143, %specific_fn.loc143
// CHECK:STDOUT:   %.loc143_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1 = specific_constant imports.%Core.Less.511, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1]
// CHECK:STDOUT:   %Less.ref.loc143: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1 = name_ref Less, %.loc143_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc143: <bound method> = bound_method %a.ref.loc143, %Less.ref.loc143
// CHECK:STDOUT:   %impl.elem0.loc143_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc143_5.3: <bound method> = bound_method %int_1.loc143, %impl.elem0.loc143_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_5: init %Cpp.long_long = call %bound_method.loc143_5.3(%int_1.loc143) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc143_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc143_5.4: %Cpp.long_long = converted %int_1.loc143, %.loc143_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc143: <specific function> = specific_function %Less.ref.loc143, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.2]
// CHECK:STDOUT:   %bound_method.loc143_5.4: <bound method> = bound_method %a.ref.loc143, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc143
// CHECK:STDOUT:   %impl.elem0.loc143_7: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc143_7: <bound method> = bound_method %int_1.loc143, %impl.elem0.loc143_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_7: init %Cpp.long_long = call %bound_method.loc143_7(%int_1.loc143) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc143_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc143_7.2: %Cpp.long_long = converted %int_1.loc143, %.loc143_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc143: init bool = call %bound_method.loc143_5.4(%a.ref.loc143, %.loc143_7.2)
// CHECK:STDOUT:   %a.ref.loc144: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc144: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc144: %.ea1 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.2]
// CHECK:STDOUT:   %bound_method.loc144_5.1: <bound method> = bound_method %a.ref.loc144, %impl.elem3.loc144
// CHECK:STDOUT:   %ImplicitAs.facet.loc144: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc144_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc144 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc144: <specific function> = specific_function %impl.elem3.loc144, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.1]
// CHECK:STDOUT:   %bound_method.loc144_5.2: <bound method> = bound_method %a.ref.loc144, %specific_fn.loc144
// CHECK:STDOUT:   %.loc144_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1 = specific_constant imports.%Core.GreaterOrEquivalent.489, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc144: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1 = name_ref GreaterOrEquivalent, %.loc144_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc144: <bound method> = bound_method %a.ref.loc144, %GreaterOrEquivalent.ref.loc144
// CHECK:STDOUT:   %impl.elem0.loc144_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc144_5.3: <bound method> = bound_method %int_1.loc144, %impl.elem0.loc144_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_5: init %Cpp.long_long = call %bound_method.loc144_5.3(%int_1.loc144) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc144_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc144_5.4: %Cpp.long_long = converted %int_1.loc144, %.loc144_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc144: <specific function> = specific_function %GreaterOrEquivalent.ref.loc144, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.2]
// CHECK:STDOUT:   %bound_method.loc144_5.4: <bound method> = bound_method %a.ref.loc144, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc144
// CHECK:STDOUT:   %impl.elem0.loc144_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc144_8: <bound method> = bound_method %int_1.loc144, %impl.elem0.loc144_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_8: init %Cpp.long_long = call %bound_method.loc144_8(%int_1.loc144) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc144_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc144_8.2: %Cpp.long_long = converted %int_1.loc144, %.loc144_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc144: init bool = call %bound_method.loc144_5.4(%a.ref.loc144, %.loc144_8.2)
// CHECK:STDOUT:   %a.ref.loc145: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc145: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc145: %.da4 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.2]
// CHECK:STDOUT:   %bound_method.loc145_5.1: <bound method> = bound_method %a.ref.loc145, %impl.elem1.loc145
// CHECK:STDOUT:   %ImplicitAs.facet.loc145: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc145_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc145 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc145: <specific function> = specific_function %impl.elem1.loc145, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.1]
// CHECK:STDOUT:   %bound_method.loc145_5.2: <bound method> = bound_method %a.ref.loc145, %specific_fn.loc145
// CHECK:STDOUT:   %.loc145_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1 = specific_constant imports.%Core.LessOrEquivalent.aaf, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc145: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1 = name_ref LessOrEquivalent, %.loc145_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc145: <bound method> = bound_method %a.ref.loc145, %LessOrEquivalent.ref.loc145
// CHECK:STDOUT:   %impl.elem0.loc145_5: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc145_5.3: <bound method> = bound_method %int_1.loc145, %impl.elem0.loc145_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_5: init %Cpp.long_long = call %bound_method.loc145_5.3(%int_1.loc145) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc145_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc145_5.4: %Cpp.long_long = converted %int_1.loc145, %.loc145_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc145: <specific function> = specific_function %LessOrEquivalent.ref.loc145, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.2]
// CHECK:STDOUT:   %bound_method.loc145_5.4: <bound method> = bound_method %a.ref.loc145, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc145
// CHECK:STDOUT:   %impl.elem0.loc145_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc145_8: <bound method> = bound_method %int_1.loc145, %impl.elem0.loc145_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_8: init %Cpp.long_long = call %bound_method.loc145_8(%int_1.loc145) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc145_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc145_8.2: %Cpp.long_long = converted %int_1.loc145, %.loc145_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc145: init bool = call %bound_method.loc145_5.4(%a.ref.loc145, %.loc145_8.2)
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
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.3, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.4, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.3, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bbf235.1: %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.4, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bbf235.2: %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.52e: <witness> = impl_witness imports.%EqWith.impl_witness_table.7e5, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.4, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.971403.1: %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.4, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.3, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.971403.2: %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.3, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.71f: %EqWith.type.4be = facet_value %i64, (%EqWith.impl_witness.52e) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.7b1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.71f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.dfc: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.71f) [concrete]
// CHECK:STDOUT:   %.692: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.7b1, %EqWith.facet.71f [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.14854f.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.971403.2, @T.binding.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.14854f.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.971403.1, @T.binding.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.1ce: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.dfc, %EqWith.facet.71f [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.b0f056.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.2, @T.binding.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.b0f056.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.1, @T.binding.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.3, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.4, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.3, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.31696a.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.4, @T.binding.as_type.as.OrderedWith.impl.895(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.31696a.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.5fb: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.8cd, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.4846a7.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.356766.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.4846a7.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.356766.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.2d0: %OrderedWith.type.65e = facet_value %i64, (%OrderedWith.impl_witness.5fb) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ee5: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.2d0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.9f3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.2d0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c5a: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.2d0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.3ba: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.2d0) [concrete]
// CHECK:STDOUT:   %.c4a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.c5a, %OrderedWith.facet.2d0 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.fc67f6.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.356766.2, @T.binding.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.fc67f6.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.356766.1, @T.binding.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.468: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ee5, %OrderedWith.facet.2d0 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.ab0306.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.4846a7.2, @T.binding.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.ab0306.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.4846a7.1, @T.binding.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.a3a: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.3ba, %OrderedWith.facet.2d0 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b93d5d.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b93d5d.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.ee9: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.9f3, %OrderedWith.facet.2d0 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.67339b.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.2, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.67339b.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.1, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.a59: <witness> = impl_witness imports.%EqWith.impl_witness_table.7e5, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.4, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.4, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.3, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.3, @T.binding.as_type.as.EqWith.impl.b07(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.db2: %EqWith.type.4be = facet_value Core.IntLiteral, (%EqWith.impl_witness.a59) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.634: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.db2) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.297: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.db2) [concrete]
// CHECK:STDOUT:   %.b72: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.634, %EqWith.facet.db2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2, @T.binding.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.1699f5.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1, @T.binding.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.1699f5.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2 [concrete]
// CHECK:STDOUT:   %.c36: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.297, %EqWith.facet.db2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.2, @T.binding.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.7aede5.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1, @T.binding.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.7aede5.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.aae: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.8cd, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.binding.as_type.as.OrderedWith.impl.895(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.88a: %OrderedWith.type.65e = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.aae) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.d22: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.88a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.2d9: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.88a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.3ab: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.88a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.907: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.88a) [concrete]
// CHECK:STDOUT:   %.dee: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.3ab, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2, @T.binding.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.a32da1.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1, @T.binding.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.a32da1.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2 [concrete]
// CHECK:STDOUT:   %.97f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.d22, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2, @T.binding.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.8f808d.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1, @T.binding.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.8f808d.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2 [concrete]
// CHECK:STDOUT:   %.d57: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.907, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.2c548f.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.2c548f.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2 [concrete]
// CHECK:STDOUT:   %.ba2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.2d9, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.caef99.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.caef99.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.eb5: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.type.2 (%T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.2 (constants.%T.binding.as_type.as.EqWith.impl.Equal.bbf235.1)]
// CHECK:STDOUT:   %Core.import_ref.597: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.type.2 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.7e5 = impl_witness_table (%Core.import_ref.eb5, %Core.import_ref.597), @T.binding.as_type.as.EqWith.impl.b07 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.421: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.type.1 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.2)]
// CHECK:STDOUT:   %Core.Equal.d94: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.type.1 (%T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.1 (constants.%T.binding.as_type.as.EqWith.impl.Equal.bbf235.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d4b: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.type.2 (%T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.31696a.1)]
// CHECK:STDOUT:   %Core.import_ref.54d: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.1)]
// CHECK:STDOUT:   %Core.import_ref.aa8: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.type.2 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.1)]
// CHECK:STDOUT:   %Core.import_ref.240: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.8cd = impl_witness_table (%Core.import_ref.d4b, %Core.import_ref.54d, %Core.import_ref.aa8, %Core.import_ref.240), @T.binding.as_type.as.OrderedWith.impl.895 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.b91: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.2)]
// CHECK:STDOUT:   %Core.Greater.cff: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.type.1 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.792: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.2)]
// CHECK:STDOUT:   %Core.Less.aac: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.type.1 (%T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.31696a.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc158: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc158_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc158: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc158: <bound method> = bound_method %int_1.loc158, %impl.elem0.loc158 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc158: init %Cpp.long_long = call %bound_method.loc158(%int_1.loc158) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc158_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc158 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc158_26.2: %Cpp.long_long = converted %int_1.loc158, %.loc158_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc158_26.2
// CHECK:STDOUT:   %b.ref.loc159: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc159: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc159_5: %.692 = impl_witness_access constants.%EqWith.impl_witness.52e, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.971403.2]
// CHECK:STDOUT:   %bound_method.loc159_5.1: <bound method> = bound_method %b.ref.loc159, %impl.elem0.loc159_5
// CHECK:STDOUT:   %specific_fn.loc159: <specific function> = specific_function %impl.elem0.loc159_5, @T.binding.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.14854f.1]
// CHECK:STDOUT:   %bound_method.loc159_5.2: <bound method> = bound_method %b.ref.loc159, %specific_fn.loc159
// CHECK:STDOUT:   %.loc159_5: %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.1 = specific_constant imports.%Core.Equal.d94, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.971403.1]
// CHECK:STDOUT:   %Equal.ref.loc159: %T.binding.as_type.as.EqWith.impl.Equal.type.5ab1af.1 = name_ref Equal, %.loc159_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.971403.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc159: <bound method> = bound_method %b.ref.loc159, %Equal.ref.loc159
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc159: <specific function> = specific_function %Equal.ref.loc159, @T.binding.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.14854f.2]
// CHECK:STDOUT:   %bound_method.loc159_5.3: <bound method> = bound_method %b.ref.loc159, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc159
// CHECK:STDOUT:   %impl.elem0.loc159_3: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc159_3: <bound method> = bound_method %b.ref.loc159, %impl.elem0.loc159_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc159: init %Cpp.long_long = call %bound_method.loc159_3(%b.ref.loc159)
// CHECK:STDOUT:   %.loc159_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc159
// CHECK:STDOUT:   %.loc159_3.2: %Cpp.long_long = converted %b.ref.loc159, %.loc159_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc159: init bool = call %bound_method.loc159_5.3(%.loc159_3.2, %a.ref.loc159)
// CHECK:STDOUT:   %b.ref.loc160: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc160: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc160: %.1ce = impl_witness_access constants.%EqWith.impl_witness.52e, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.2]
// CHECK:STDOUT:   %bound_method.loc160_5.1: <bound method> = bound_method %b.ref.loc160, %impl.elem1.loc160
// CHECK:STDOUT:   %specific_fn.loc160: <specific function> = specific_function %impl.elem1.loc160, @T.binding.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.b0f056.1]
// CHECK:STDOUT:   %bound_method.loc160_5.2: <bound method> = bound_method %b.ref.loc160, %specific_fn.loc160
// CHECK:STDOUT:   %.loc160_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.1 = specific_constant imports.%Core.NotEqual.421, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.1]
// CHECK:STDOUT:   %NotEqual.ref.loc160: %T.binding.as_type.as.EqWith.impl.NotEqual.type.7cb54d.1 = name_ref NotEqual, %.loc160_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bd87a0.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc160: <bound method> = bound_method %b.ref.loc160, %NotEqual.ref.loc160
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc160: <specific function> = specific_function %NotEqual.ref.loc160, @T.binding.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.b0f056.2]
// CHECK:STDOUT:   %bound_method.loc160_5.3: <bound method> = bound_method %b.ref.loc160, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc160
// CHECK:STDOUT:   %impl.elem0.loc160: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc160_3: <bound method> = bound_method %b.ref.loc160, %impl.elem0.loc160
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc160: init %Cpp.long_long = call %bound_method.loc160_3(%b.ref.loc160)
// CHECK:STDOUT:   %.loc160_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc160
// CHECK:STDOUT:   %.loc160_3.2: %Cpp.long_long = converted %b.ref.loc160, %.loc160_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc160: init bool = call %bound_method.loc160_5.3(%.loc160_3.2, %a.ref.loc160)
// CHECK:STDOUT:   %b.ref.loc161: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc161: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc161: %.c4a = impl_witness_access constants.%OrderedWith.impl_witness.5fb, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.356766.2]
// CHECK:STDOUT:   %bound_method.loc161_5.1: <bound method> = bound_method %b.ref.loc161, %impl.elem2.loc161
// CHECK:STDOUT:   %specific_fn.loc161: <specific function> = specific_function %impl.elem2.loc161, @T.binding.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.fc67f6.1]
// CHECK:STDOUT:   %bound_method.loc161_5.2: <bound method> = bound_method %b.ref.loc161, %specific_fn.loc161
// CHECK:STDOUT:   %.loc161_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.1 = specific_constant imports.%Core.Greater.cff, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.356766.1]
// CHECK:STDOUT:   %Greater.ref.loc161: %T.binding.as_type.as.OrderedWith.impl.Greater.type.125113.1 = name_ref Greater, %.loc161_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.356766.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc161: <bound method> = bound_method %b.ref.loc161, %Greater.ref.loc161
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc161: <specific function> = specific_function %Greater.ref.loc161, @T.binding.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.fc67f6.2]
// CHECK:STDOUT:   %bound_method.loc161_5.3: <bound method> = bound_method %b.ref.loc161, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc161
// CHECK:STDOUT:   %impl.elem0.loc161: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc161_3: <bound method> = bound_method %b.ref.loc161, %impl.elem0.loc161
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc161: init %Cpp.long_long = call %bound_method.loc161_3(%b.ref.loc161)
// CHECK:STDOUT:   %.loc161_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc161
// CHECK:STDOUT:   %.loc161_3.2: %Cpp.long_long = converted %b.ref.loc161, %.loc161_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc161: init bool = call %bound_method.loc161_5.3(%.loc161_3.2, %a.ref.loc161)
// CHECK:STDOUT:   %b.ref.loc162: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc162: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc162_5: %.468 = impl_witness_access constants.%OrderedWith.impl_witness.5fb, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.4846a7.2]
// CHECK:STDOUT:   %bound_method.loc162_5.1: <bound method> = bound_method %b.ref.loc162, %impl.elem0.loc162_5
// CHECK:STDOUT:   %specific_fn.loc162: <specific function> = specific_function %impl.elem0.loc162_5, @T.binding.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.ab0306.1]
// CHECK:STDOUT:   %bound_method.loc162_5.2: <bound method> = bound_method %b.ref.loc162, %specific_fn.loc162
// CHECK:STDOUT:   %.loc162_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.1 = specific_constant imports.%Core.Less.aac, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.4846a7.1]
// CHECK:STDOUT:   %Less.ref.loc162: %T.binding.as_type.as.OrderedWith.impl.Less.type.8c7c29.1 = name_ref Less, %.loc162_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.4846a7.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc162: <bound method> = bound_method %b.ref.loc162, %Less.ref.loc162
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc162: <specific function> = specific_function %Less.ref.loc162, @T.binding.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.ab0306.2]
// CHECK:STDOUT:   %bound_method.loc162_5.3: <bound method> = bound_method %b.ref.loc162, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc162
// CHECK:STDOUT:   %impl.elem0.loc162_3: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc162_3: <bound method> = bound_method %b.ref.loc162, %impl.elem0.loc162_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc162: init %Cpp.long_long = call %bound_method.loc162_3(%b.ref.loc162)
// CHECK:STDOUT:   %.loc162_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc162
// CHECK:STDOUT:   %.loc162_3.2: %Cpp.long_long = converted %b.ref.loc162, %.loc162_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc162: init bool = call %bound_method.loc162_5.3(%.loc162_3.2, %a.ref.loc162)
// CHECK:STDOUT:   %b.ref.loc163: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc163: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc163: %.a3a = impl_witness_access constants.%OrderedWith.impl_witness.5fb, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.2]
// CHECK:STDOUT:   %bound_method.loc163_5.1: <bound method> = bound_method %b.ref.loc163, %impl.elem3.loc163
// CHECK:STDOUT:   %specific_fn.loc163: <specific function> = specific_function %impl.elem3.loc163, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b93d5d.1]
// CHECK:STDOUT:   %bound_method.loc163_5.2: <bound method> = bound_method %b.ref.loc163, %specific_fn.loc163
// CHECK:STDOUT:   %.loc163_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.1 = specific_constant imports.%Core.GreaterOrEquivalent.b91, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc163: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6d7684.1 = name_ref GreaterOrEquivalent, %.loc163_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7a66f5.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc163: <bound method> = bound_method %b.ref.loc163, %GreaterOrEquivalent.ref.loc163
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc163: <specific function> = specific_function %GreaterOrEquivalent.ref.loc163, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.b93d5d.2]
// CHECK:STDOUT:   %bound_method.loc163_5.3: <bound method> = bound_method %b.ref.loc163, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc163
// CHECK:STDOUT:   %impl.elem0.loc163: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc163_3: <bound method> = bound_method %b.ref.loc163, %impl.elem0.loc163
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc163: init %Cpp.long_long = call %bound_method.loc163_3(%b.ref.loc163)
// CHECK:STDOUT:   %.loc163_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc163
// CHECK:STDOUT:   %.loc163_3.2: %Cpp.long_long = converted %b.ref.loc163, %.loc163_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc163: init bool = call %bound_method.loc163_5.3(%.loc163_3.2, %a.ref.loc163)
// CHECK:STDOUT:   %b.ref.loc164: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc164: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc164: %.ee9 = impl_witness_access constants.%OrderedWith.impl_witness.5fb, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.2]
// CHECK:STDOUT:   %bound_method.loc164_5.1: <bound method> = bound_method %b.ref.loc164, %impl.elem1.loc164
// CHECK:STDOUT:   %specific_fn.loc164: <specific function> = specific_function %impl.elem1.loc164, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.67339b.1]
// CHECK:STDOUT:   %bound_method.loc164_5.2: <bound method> = bound_method %b.ref.loc164, %specific_fn.loc164
// CHECK:STDOUT:   %.loc164_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.1 = specific_constant imports.%Core.LessOrEquivalent.792, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc164: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bba1e4.1 = name_ref LessOrEquivalent, %.loc164_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.ebfce3.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc164: <bound method> = bound_method %b.ref.loc164, %LessOrEquivalent.ref.loc164
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc164: <specific function> = specific_function %LessOrEquivalent.ref.loc164, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.c59) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.67339b.2]
// CHECK:STDOUT:   %bound_method.loc164_5.3: <bound method> = bound_method %b.ref.loc164, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc164
// CHECK:STDOUT:   %impl.elem0.loc164: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc164_3: <bound method> = bound_method %b.ref.loc164, %impl.elem0.loc164
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc164: init %Cpp.long_long = call %bound_method.loc164_3(%b.ref.loc164)
// CHECK:STDOUT:   %.loc164_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc164
// CHECK:STDOUT:   %.loc164_3.2: %Cpp.long_long = converted %b.ref.loc164, %.loc164_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc164: init bool = call %bound_method.loc164_5.3(%.loc164_3.2, %a.ref.loc164)
// CHECK:STDOUT:   %int_1.loc166: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc166: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc166_5: %.b72 = impl_witness_access constants.%EqWith.impl_witness.a59, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2]
// CHECK:STDOUT:   %bound_method.loc166_5.1: <bound method> = bound_method %int_1.loc166, %impl.elem0.loc166_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.1]
// CHECK:STDOUT:   %specific_fn.loc166: <specific function> = specific_function %impl.elem0.loc166_5, @T.binding.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1]
// CHECK:STDOUT:   %bound_method.loc166_5.2: <bound method> = bound_method %int_1.loc166, %specific_fn.loc166 [concrete = constants.%bound_method.1699f5.1]
// CHECK:STDOUT:   %.loc166_5: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1 = specific_constant imports.%Core.Equal.d94, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1]
// CHECK:STDOUT:   %Equal.ref.loc166: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1 = name_ref Equal, %.loc166_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc166: <bound method> = bound_method %int_1.loc166, %Equal.ref.loc166 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc166: <specific function> = specific_function %Equal.ref.loc166, @T.binding.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2]
// CHECK:STDOUT:   %bound_method.loc166_5.3: <bound method> = bound_method %int_1.loc166, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc166 [concrete = constants.%bound_method.1699f5.2]
// CHECK:STDOUT:   %impl.elem0.loc166_3: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc166_3: <bound method> = bound_method %int_1.loc166, %impl.elem0.loc166_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc166: init %Cpp.long_long = call %bound_method.loc166_3(%int_1.loc166) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc166_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc166 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc166_3.2: %Cpp.long_long = converted %int_1.loc166, %.loc166_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc166: init bool = call %bound_method.loc166_5.3(%.loc166_3.2, %a.ref.loc166)
// CHECK:STDOUT:   %int_1.loc167: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc167: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc167: %.c36 = impl_witness_access constants.%EqWith.impl_witness.a59, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.2]
// CHECK:STDOUT:   %bound_method.loc167_5.1: <bound method> = bound_method %int_1.loc167, %impl.elem1.loc167 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.1]
// CHECK:STDOUT:   %specific_fn.loc167: <specific function> = specific_function %impl.elem1.loc167, @T.binding.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.1]
// CHECK:STDOUT:   %bound_method.loc167_5.2: <bound method> = bound_method %int_1.loc167, %specific_fn.loc167 [concrete = constants.%bound_method.7aede5.1]
// CHECK:STDOUT:   %.loc167_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1 = specific_constant imports.%Core.NotEqual.421, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1]
// CHECK:STDOUT:   %NotEqual.ref.loc167: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1 = name_ref NotEqual, %.loc167_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc167: <bound method> = bound_method %int_1.loc167, %NotEqual.ref.loc167 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc167: <specific function> = specific_function %NotEqual.ref.loc167, @T.binding.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.2]
// CHECK:STDOUT:   %bound_method.loc167_5.3: <bound method> = bound_method %int_1.loc167, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc167 [concrete = constants.%bound_method.7aede5.2]
// CHECK:STDOUT:   %impl.elem0.loc167: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc167_3: <bound method> = bound_method %int_1.loc167, %impl.elem0.loc167 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc167: init %Cpp.long_long = call %bound_method.loc167_3(%int_1.loc167) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc167_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc167 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc167_3.2: %Cpp.long_long = converted %int_1.loc167, %.loc167_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc167: init bool = call %bound_method.loc167_5.3(%.loc167_3.2, %a.ref.loc167)
// CHECK:STDOUT:   %int_1.loc168: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc168: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc168: %.dee = impl_witness_access constants.%OrderedWith.impl_witness.aae, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2]
// CHECK:STDOUT:   %bound_method.loc168_5.1: <bound method> = bound_method %int_1.loc168, %impl.elem2.loc168 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.1]
// CHECK:STDOUT:   %specific_fn.loc168: <specific function> = specific_function %impl.elem2.loc168, @T.binding.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1]
// CHECK:STDOUT:   %bound_method.loc168_5.2: <bound method> = bound_method %int_1.loc168, %specific_fn.loc168 [concrete = constants.%bound_method.a32da1.1]
// CHECK:STDOUT:   %.loc168_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1 = specific_constant imports.%Core.Greater.cff, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1]
// CHECK:STDOUT:   %Greater.ref.loc168: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1 = name_ref Greater, %.loc168_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc168: <bound method> = bound_method %int_1.loc168, %Greater.ref.loc168 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc168: <specific function> = specific_function %Greater.ref.loc168, @T.binding.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2]
// CHECK:STDOUT:   %bound_method.loc168_5.3: <bound method> = bound_method %int_1.loc168, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc168 [concrete = constants.%bound_method.a32da1.2]
// CHECK:STDOUT:   %impl.elem0.loc168: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc168_3: <bound method> = bound_method %int_1.loc168, %impl.elem0.loc168 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc168: init %Cpp.long_long = call %bound_method.loc168_3(%int_1.loc168) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc168_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc168 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc168_3.2: %Cpp.long_long = converted %int_1.loc168, %.loc168_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc168: init bool = call %bound_method.loc168_5.3(%.loc168_3.2, %a.ref.loc168)
// CHECK:STDOUT:   %int_1.loc169: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc169: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc169_5: %.97f = impl_witness_access constants.%OrderedWith.impl_witness.aae, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2]
// CHECK:STDOUT:   %bound_method.loc169_5.1: <bound method> = bound_method %int_1.loc169, %impl.elem0.loc169_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.1]
// CHECK:STDOUT:   %specific_fn.loc169: <specific function> = specific_function %impl.elem0.loc169_5, @T.binding.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1]
// CHECK:STDOUT:   %bound_method.loc169_5.2: <bound method> = bound_method %int_1.loc169, %specific_fn.loc169 [concrete = constants.%bound_method.8f808d.1]
// CHECK:STDOUT:   %.loc169_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1 = specific_constant imports.%Core.Less.aac, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1]
// CHECK:STDOUT:   %Less.ref.loc169: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1 = name_ref Less, %.loc169_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc169: <bound method> = bound_method %int_1.loc169, %Less.ref.loc169 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc169: <specific function> = specific_function %Less.ref.loc169, @T.binding.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2]
// CHECK:STDOUT:   %bound_method.loc169_5.3: <bound method> = bound_method %int_1.loc169, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc169 [concrete = constants.%bound_method.8f808d.2]
// CHECK:STDOUT:   %impl.elem0.loc169_3: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc169_3: <bound method> = bound_method %int_1.loc169, %impl.elem0.loc169_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc169: init %Cpp.long_long = call %bound_method.loc169_3(%int_1.loc169) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc169_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc169 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc169_3.2: %Cpp.long_long = converted %int_1.loc169, %.loc169_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc169: init bool = call %bound_method.loc169_5.3(%.loc169_3.2, %a.ref.loc169)
// CHECK:STDOUT:   %int_1.loc170: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc170: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc170: %.d57 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2]
// CHECK:STDOUT:   %bound_method.loc170_5.1: <bound method> = bound_method %int_1.loc170, %impl.elem3.loc170 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.1]
// CHECK:STDOUT:   %specific_fn.loc170: <specific function> = specific_function %impl.elem3.loc170, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1]
// CHECK:STDOUT:   %bound_method.loc170_5.2: <bound method> = bound_method %int_1.loc170, %specific_fn.loc170 [concrete = constants.%bound_method.2c548f.1]
// CHECK:STDOUT:   %.loc170_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1 = specific_constant imports.%Core.GreaterOrEquivalent.b91, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc170: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1 = name_ref GreaterOrEquivalent, %.loc170_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc170: <bound method> = bound_method %int_1.loc170, %GreaterOrEquivalent.ref.loc170 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc170: <specific function> = specific_function %GreaterOrEquivalent.ref.loc170, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2]
// CHECK:STDOUT:   %bound_method.loc170_5.3: <bound method> = bound_method %int_1.loc170, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc170 [concrete = constants.%bound_method.2c548f.2]
// CHECK:STDOUT:   %impl.elem0.loc170: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc170_3: <bound method> = bound_method %int_1.loc170, %impl.elem0.loc170 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc170: init %Cpp.long_long = call %bound_method.loc170_3(%int_1.loc170) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc170_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc170 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc170_3.2: %Cpp.long_long = converted %int_1.loc170, %.loc170_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc170: init bool = call %bound_method.loc170_5.3(%.loc170_3.2, %a.ref.loc170)
// CHECK:STDOUT:   %int_1.loc171: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc171: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc171: %.ba2 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2]
// CHECK:STDOUT:   %bound_method.loc171_5.1: <bound method> = bound_method %int_1.loc171, %impl.elem1.loc171 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.1]
// CHECK:STDOUT:   %specific_fn.loc171: <specific function> = specific_function %impl.elem1.loc171, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1]
// CHECK:STDOUT:   %bound_method.loc171_5.2: <bound method> = bound_method %int_1.loc171, %specific_fn.loc171 [concrete = constants.%bound_method.caef99.1]
// CHECK:STDOUT:   %.loc171_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1 = specific_constant imports.%Core.LessOrEquivalent.792, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc171: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1 = name_ref LessOrEquivalent, %.loc171_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc171: <bound method> = bound_method %int_1.loc171, %LessOrEquivalent.ref.loc171 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc171: <specific function> = specific_function %LessOrEquivalent.ref.loc171, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2]
// CHECK:STDOUT:   %bound_method.loc171_5.3: <bound method> = bound_method %int_1.loc171, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc171 [concrete = constants.%bound_method.caef99.2]
// CHECK:STDOUT:   %impl.elem0.loc171: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc171_3: <bound method> = bound_method %int_1.loc171, %impl.elem0.loc171 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc171: init %Cpp.long_long = call %bound_method.loc171_3(%int_1.loc171) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc171_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc171 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc171_3.2: %Cpp.long_long = converted %int_1.loc171, %.loc171_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc171: init bool = call %bound_method.loc171_5.3(%.loc171_3.2, %a.ref.loc171)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert, @T.binding.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.19b: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.6d4: <witness> = impl_witness imports.%Copy.impl_witness_table.804 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.6d4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.64f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f1e, @T.binding.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c8d: %ImplicitAs.type.a03 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.64f) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.383: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.815, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.2ec = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.383) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.815: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.4a9: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.815, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.2b826b.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.2b826b.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b35: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.e66: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.705, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.07d295.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.07d295.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b35 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.e66) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.55e: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.858: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.55e, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.21e13f.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.07d295.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.21e13f.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.07d295.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.112: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.04a: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.32a, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.112 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.04a) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3c8: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.930: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3c8, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.5af50b.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.5af50b.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.45d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.12b589.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.12b589.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.5ff: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.9c9, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.45d = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.5ff) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.f6a: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.1ba: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.f6a, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.07c8ad.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.07c8ad.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.a2e: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.afa: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.fb0, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.a2e = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.afa) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.4e1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9bd: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.4e1, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.a3301b.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.a3301b.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.9bc: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.e8f: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.5cc, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.9bc = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.e8f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.a11: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e9f: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.a11, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.5df9d8.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.5df9d8.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.219: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.3af: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.a52, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.219 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.3af) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.9e0: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.22a: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.9e0, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.857890.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.857890.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.f9d: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.56d: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.223, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.f9d = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.56d) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.d25: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.270: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.d25, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.97edcc.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.97edcc.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.f24: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.824: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.cd9, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.f24 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.824) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.cf9: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ecd: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.cf9, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.525b00.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.525b00.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.42e: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.d22: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.074, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.42e = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.d22) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.eb3: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.8eb: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.eb3, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.df86ad.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.df86ad.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c8d) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1c6: @T.binding.as_type.as.ImplicitAs.impl.%T.binding.as_type.as.ImplicitAs.impl.Convert.type (%T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.binding.as_type.as.ImplicitAs.impl.%T.binding.as_type.as.ImplicitAs.impl.Convert (constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.19b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f1e = impl_witness_table (%Core.import_ref.1c6), @T.binding.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.277: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.815 = impl_witness_table (%Core.import_ref.277), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.57a: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2)]
// CHECK:STDOUT:   %Core.import_ref.7a9: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.804 = impl_witness_table (%Core.import_ref.7a9), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.819: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.705 = impl_witness_table (%Core.import_ref.819), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.75b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.2)]
// CHECK:STDOUT:   %Core.import_ref.f92: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.32a = impl_witness_table (%Core.import_ref.f92), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.94e: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.2)]
// CHECK:STDOUT:   %Core.import_ref.b94: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.12b589.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.9c9 = impl_witness_table (%Core.import_ref.b94), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ec7: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.12b589.2)]
// CHECK:STDOUT:   %Core.import_ref.797: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.797), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6fd: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.2)]
// CHECK:STDOUT:   %Core.import_ref.e22: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.5cc = impl_witness_table (%Core.import_ref.e22), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c78: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.2)]
// CHECK:STDOUT:   %Core.import_ref.460: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.a52 = impl_witness_table (%Core.import_ref.460), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fa0: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.2)]
// CHECK:STDOUT:   %Core.import_ref.bb9: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.223 = impl_witness_table (%Core.import_ref.bb9), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3d4: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.2)]
// CHECK:STDOUT:   %Core.import_ref.db6: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.cd9 = impl_witness_table (%Core.import_ref.db6), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b6: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.2)]
// CHECK:STDOUT:   %Core.import_ref.522: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.074 = impl_witness_table (%Core.import_ref.522), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.bcc: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc456: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc456: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc456: <bound method> = bound_method %int_1.loc456, %impl.elem0.loc456 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc456: init %Cpp.long_long = call %bound_method.loc456(%int_1.loc456) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc456_3: init %Cpp.long_long = converted %int_1.loc456, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc456 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc456_3
// CHECK:STDOUT:   %.loc456_13: type = splice_block %long_long.ref.loc456 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc456: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc456: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.76e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.76e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc457: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc457: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc457: <bound method> = bound_method %int_1.loc457, %impl.elem0.loc457 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc457: init %Cpp.long_long = call %bound_method.loc457(%int_1.loc457) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc457_3: init %Cpp.long_long = converted %int_1.loc457, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc457 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %b.var, %.loc457_3
// CHECK:STDOUT:   %.loc457_13: type = splice_block %long_long.ref.loc457 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc457: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc457: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc459: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc459: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc459: %.4a9 = impl_witness_access constants.%AddAssignWith.impl_witness.383, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.2]
// CHECK:STDOUT:   %bound_method.loc459_5.1: <bound method> = bound_method %a.ref.loc459, %impl.elem0.loc459
// CHECK:STDOUT:   %ImplicitAs.facet.loc459: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc459_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc459 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc459: <specific function> = specific_function %impl.elem0.loc459, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.2b826b.1]
// CHECK:STDOUT:   %bound_method.loc459_5.2: <bound method> = bound_method %a.ref.loc459, %specific_fn.loc459
// CHECK:STDOUT:   %.loc459_8: %Cpp.long_long = acquire_value %b.ref.loc459
// CHECK:STDOUT:   %.loc459_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.1 = specific_constant imports.%Core.Op.57a, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.1]
// CHECK:STDOUT:   %Op.ref.loc459: %Cpp.long_long.as.AddAssignWith.impl.Op.type.26466a.1 = name_ref Op, %.loc459_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d128e4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc459, %Op.ref.loc459
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc459, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.2b826b.2]
// CHECK:STDOUT:   %bound_method.loc459_5.3: <bound method> = bound_method %a.ref.loc459, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc459_5.3(%a.ref.loc459, %.loc459_8)
// CHECK:STDOUT:   %a.ref.loc460: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc460: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc460: %.858 = impl_witness_access constants.%SubAssignWith.impl_witness.e66, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.07d295.2]
// CHECK:STDOUT:   %bound_method.loc460_5.1: <bound method> = bound_method %a.ref.loc460, %impl.elem0.loc460
// CHECK:STDOUT:   %ImplicitAs.facet.loc460: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc460_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc460 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc460: <specific function> = specific_function %impl.elem0.loc460, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.21e13f.1]
// CHECK:STDOUT:   %bound_method.loc460_5.2: <bound method> = bound_method %a.ref.loc460, %specific_fn.loc460
// CHECK:STDOUT:   %.loc460_8: %Cpp.long_long = acquire_value %b.ref.loc460
// CHECK:STDOUT:   %.loc460_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.1 = specific_constant imports.%Core.Op.75b, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.07d295.1]
// CHECK:STDOUT:   %Op.ref.loc460: %Cpp.long_long.as.SubAssignWith.impl.Op.type.daa062.1 = name_ref Op, %.loc460_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.07d295.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc460, %Op.ref.loc460
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc460, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.21e13f.2]
// CHECK:STDOUT:   %bound_method.loc460_5.3: <bound method> = bound_method %a.ref.loc460, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc460_5.3(%a.ref.loc460, %.loc460_8)
// CHECK:STDOUT:   %a.ref.loc461: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc461: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc461: %.930 = impl_witness_access constants.%MulAssignWith.impl_witness.04a, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.2]
// CHECK:STDOUT:   %bound_method.loc461_5.1: <bound method> = bound_method %a.ref.loc461, %impl.elem0.loc461
// CHECK:STDOUT:   %ImplicitAs.facet.loc461: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc461_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc461 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc461: <specific function> = specific_function %impl.elem0.loc461, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.5af50b.1]
// CHECK:STDOUT:   %bound_method.loc461_5.2: <bound method> = bound_method %a.ref.loc461, %specific_fn.loc461
// CHECK:STDOUT:   %.loc461_8: %Cpp.long_long = acquire_value %b.ref.loc461
// CHECK:STDOUT:   %.loc461_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.1 = specific_constant imports.%Core.Op.94e, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.1]
// CHECK:STDOUT:   %Op.ref.loc461: %Cpp.long_long.as.MulAssignWith.impl.Op.type.f5880d.1 = name_ref Op, %.loc461_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.0edfbc.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc461, %Op.ref.loc461
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc461, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.5af50b.2]
// CHECK:STDOUT:   %bound_method.loc461_5.3: <bound method> = bound_method %a.ref.loc461, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc461_5.3(%a.ref.loc461, %.loc461_8)
// CHECK:STDOUT:   %a.ref.loc462: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc462: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc462: %.1ba = impl_witness_access constants.%DivAssignWith.impl_witness.5ff, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.2]
// CHECK:STDOUT:   %bound_method.loc462_5.1: <bound method> = bound_method %a.ref.loc462, %impl.elem0.loc462
// CHECK:STDOUT:   %ImplicitAs.facet.loc462: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc462_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc462 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc462: <specific function> = specific_function %impl.elem0.loc462, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.07c8ad.1]
// CHECK:STDOUT:   %bound_method.loc462_5.2: <bound method> = bound_method %a.ref.loc462, %specific_fn.loc462
// CHECK:STDOUT:   %.loc462_8: %Cpp.long_long = acquire_value %b.ref.loc462
// CHECK:STDOUT:   %.loc462_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.1 = specific_constant imports.%Core.Op.ec7, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.1]
// CHECK:STDOUT:   %Op.ref.loc462: %Cpp.long_long.as.DivAssignWith.impl.Op.type.fa44cf.1 = name_ref Op, %.loc462_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.409fa6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc462, %Op.ref.loc462
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc462, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.07c8ad.2]
// CHECK:STDOUT:   %bound_method.loc462_5.3: <bound method> = bound_method %a.ref.loc462, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc462_5.3(%a.ref.loc462, %.loc462_8)
// CHECK:STDOUT:   %a.ref.loc463: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc463: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc463: %.9bd = impl_witness_access constants.%ModAssignWith.impl_witness.afa, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.2]
// CHECK:STDOUT:   %bound_method.loc463_5.1: <bound method> = bound_method %a.ref.loc463, %impl.elem0.loc463
// CHECK:STDOUT:   %ImplicitAs.facet.loc463: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc463_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc463 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc463: <specific function> = specific_function %impl.elem0.loc463, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.a3301b.1]
// CHECK:STDOUT:   %bound_method.loc463_5.2: <bound method> = bound_method %a.ref.loc463, %specific_fn.loc463
// CHECK:STDOUT:   %.loc463_8: %Cpp.long_long = acquire_value %b.ref.loc463
// CHECK:STDOUT:   %.loc463_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.1 = specific_constant imports.%Core.Op.6fd, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.1]
// CHECK:STDOUT:   %Op.ref.loc463: %Cpp.long_long.as.ModAssignWith.impl.Op.type.80b19c.1 = name_ref Op, %.loc463_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0b732b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc463, %Op.ref.loc463
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc463, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.a3301b.2]
// CHECK:STDOUT:   %bound_method.loc463_5.3: <bound method> = bound_method %a.ref.loc463, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc463_5.3(%a.ref.loc463, %.loc463_8)
// CHECK:STDOUT:   %a.ref.loc465: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc465: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc465: %.e9f = impl_witness_access constants.%BitAndAssignWith.impl_witness.e8f, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.2]
// CHECK:STDOUT:   %bound_method.loc465_5.1: <bound method> = bound_method %a.ref.loc465, %impl.elem0.loc465
// CHECK:STDOUT:   %ImplicitAs.facet.loc465: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc465_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc465 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc465: <specific function> = specific_function %impl.elem0.loc465, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.5df9d8.1]
// CHECK:STDOUT:   %bound_method.loc465_5.2: <bound method> = bound_method %a.ref.loc465, %specific_fn.loc465
// CHECK:STDOUT:   %.loc465_8: %Cpp.long_long = acquire_value %b.ref.loc465
// CHECK:STDOUT:   %.loc465_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.1 = specific_constant imports.%Core.Op.c78, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.1]
// CHECK:STDOUT:   %Op.ref.loc465: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.a7c3f6.1 = name_ref Op, %.loc465_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.10a9af.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc465, %Op.ref.loc465
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc465, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.5df9d8.2]
// CHECK:STDOUT:   %bound_method.loc465_5.3: <bound method> = bound_method %a.ref.loc465, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc465_5.3(%a.ref.loc465, %.loc465_8)
// CHECK:STDOUT:   %a.ref.loc466: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc466: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc466: %.22a = impl_witness_access constants.%BitOrAssignWith.impl_witness.3af, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.2]
// CHECK:STDOUT:   %bound_method.loc466_5.1: <bound method> = bound_method %a.ref.loc466, %impl.elem0.loc466
// CHECK:STDOUT:   %ImplicitAs.facet.loc466: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc466_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc466 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc466: <specific function> = specific_function %impl.elem0.loc466, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.857890.1]
// CHECK:STDOUT:   %bound_method.loc466_5.2: <bound method> = bound_method %a.ref.loc466, %specific_fn.loc466
// CHECK:STDOUT:   %.loc466_8: %Cpp.long_long = acquire_value %b.ref.loc466
// CHECK:STDOUT:   %.loc466_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.1 = specific_constant imports.%Core.Op.fa0, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.1]
// CHECK:STDOUT:   %Op.ref.loc466: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.367944.1 = name_ref Op, %.loc466_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.60d957.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc466, %Op.ref.loc466
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc466, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.857890.2]
// CHECK:STDOUT:   %bound_method.loc466_5.3: <bound method> = bound_method %a.ref.loc466, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc466_5.3(%a.ref.loc466, %.loc466_8)
// CHECK:STDOUT:   %a.ref.loc467: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc467: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc467: %.270 = impl_witness_access constants.%BitXorAssignWith.impl_witness.56d, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.2]
// CHECK:STDOUT:   %bound_method.loc467_5.1: <bound method> = bound_method %a.ref.loc467, %impl.elem0.loc467
// CHECK:STDOUT:   %ImplicitAs.facet.loc467: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc467_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc467 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc467: <specific function> = specific_function %impl.elem0.loc467, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.97edcc.1]
// CHECK:STDOUT:   %bound_method.loc467_5.2: <bound method> = bound_method %a.ref.loc467, %specific_fn.loc467
// CHECK:STDOUT:   %.loc467_8: %Cpp.long_long = acquire_value %b.ref.loc467
// CHECK:STDOUT:   %.loc467_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.1 = specific_constant imports.%Core.Op.3d4, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.1]
// CHECK:STDOUT:   %Op.ref.loc467: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d7a644.1 = name_ref Op, %.loc467_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.c92629.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc467, %Op.ref.loc467
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc467, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.97edcc.2]
// CHECK:STDOUT:   %bound_method.loc467_5.3: <bound method> = bound_method %a.ref.loc467, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc467_5.3(%a.ref.loc467, %.loc467_8)
// CHECK:STDOUT:   %a.ref.loc468: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc468: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc468: %.ecd = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.824, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.2]
// CHECK:STDOUT:   %bound_method.loc468_5.1: <bound method> = bound_method %a.ref.loc468, %impl.elem0.loc468
// CHECK:STDOUT:   %ImplicitAs.facet.loc468: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc468_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc468 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc468: <specific function> = specific_function %impl.elem0.loc468, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.525b00.1]
// CHECK:STDOUT:   %bound_method.loc468_5.2: <bound method> = bound_method %a.ref.loc468, %specific_fn.loc468
// CHECK:STDOUT:   %.loc468_9: %Cpp.long_long = acquire_value %b.ref.loc468
// CHECK:STDOUT:   %.loc468_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.1 = specific_constant imports.%Core.Op.1b6, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.1]
// CHECK:STDOUT:   %Op.ref.loc468: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.bbf90a.1 = name_ref Op, %.loc468_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ce8f7e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc468, %Op.ref.loc468
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc468, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.525b00.2]
// CHECK:STDOUT:   %bound_method.loc468_5.3: <bound method> = bound_method %a.ref.loc468, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc468_5.3(%a.ref.loc468, %.loc468_9)
// CHECK:STDOUT:   %a.ref.loc469: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc469: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc469: %.8eb = impl_witness_access constants.%RightShiftAssignWith.impl_witness.d22, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.2]
// CHECK:STDOUT:   %bound_method.loc469_5.1: <bound method> = bound_method %a.ref.loc469, %impl.elem0.loc469
// CHECK:STDOUT:   %ImplicitAs.facet.loc469: %ImplicitAs.type.a03 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.64f) [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %.loc469_5.1: %ImplicitAs.type.a03 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc469 [concrete = constants.%ImplicitAs.facet.c8d]
// CHECK:STDOUT:   %specific_fn.loc469: <specific function> = specific_function %impl.elem0.loc469, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.df86ad.1]
// CHECK:STDOUT:   %bound_method.loc469_5.2: <bound method> = bound_method %a.ref.loc469, %specific_fn.loc469
// CHECK:STDOUT:   %.loc469_9: %Cpp.long_long = acquire_value %b.ref.loc469
// CHECK:STDOUT:   %.loc469_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.1 = specific_constant imports.%Core.Op.bcc, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.1]
// CHECK:STDOUT:   %Op.ref.loc469: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.ff6b73.1 = name_ref Op, %.loc469_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.4e9b16.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc469, %Op.ref.loc469
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc469, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c8d) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.df86ad.2]
// CHECK:STDOUT:   %bound_method.loc469_5.3: <bound method> = bound_method %a.ref.loc469, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc469_5.3(%a.ref.loc469, %.loc469_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc457: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc457: init %empty_tuple.type = call %Destroy.Op.bound.loc457(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc456: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc456: init %empty_tuple.type = call %Destroy.Op.bound.loc456(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %AddAssignWith.type.d84: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.767: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.815, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.767) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d84: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.f7b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d84, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.d66: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.089: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.705, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.d66 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.089) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.bd6: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.b43: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.bd6, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.39aa60.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.39aa60.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.e40: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.435: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.32a, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.e40 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.435) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.dd5: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ebf: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.dd5, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.8da25e.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.8da25e.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.511: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.12b589.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.12b589.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.817: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.9c9, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.511 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.817) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.e7b: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9aa: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.e7b, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.a29bb8.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.a29bb8.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2c3: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.f7c: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.fb0, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.49994b.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.49994b.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2c3 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.f7c) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.202: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.eb2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.202, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.e3c8dd.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.49994b.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.e3c8dd.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.49994b.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.8af: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.80c: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.5cc, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.8af = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.80c) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e1b: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ba8: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.e1b, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.00c338.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.00c338.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.172: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.bd2: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.a52, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.172 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.bd2) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.21a: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.4fb: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.21a, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.b0c3cb.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.b0c3cb.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.2c8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.f91: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.223, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.2c8 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.f91) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.452: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.919: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.452, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8f3086.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8f3086.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.b97: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.c81: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.cd9, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.b97 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.c81) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.89e: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.64f: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.89e, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1c7fb8.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1c7fb8.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.721: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.1b7: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.074, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.721 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.1b7) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.eb9: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.298: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.eb9, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1c2239.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1c2239.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.277: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.815 = impl_witness_table (%Core.import_ref.277), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.57a: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2)]
// CHECK:STDOUT:   %Core.import_ref.4f4: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.819: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.705 = impl_witness_table (%Core.import_ref.819), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.75b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.092242.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.7c6ec4.2)]
// CHECK:STDOUT:   %Core.import_ref.f92: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.32a = impl_witness_table (%Core.import_ref.f92), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.94e: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.d6e66c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b4560e.2)]
// CHECK:STDOUT:   %Core.import_ref.b94: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.12b589.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.9c9 = impl_witness_table (%Core.import_ref.b94), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ec7: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.93a446.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.12b589.2)]
// CHECK:STDOUT:   %Core.import_ref.797: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.797), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6fd: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.92274d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.31c4bc.2)]
// CHECK:STDOUT:   %Core.import_ref.e22: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.5cc = impl_witness_table (%Core.import_ref.e22), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c78: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4d9ef1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.6a0d28.2)]
// CHECK:STDOUT:   %Core.import_ref.460: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.a52 = impl_witness_table (%Core.import_ref.460), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fa0: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.609f8c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1fac0c.2)]
// CHECK:STDOUT:   %Core.import_ref.bb9: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.223 = impl_witness_table (%Core.import_ref.bb9), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3d4: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.85327b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.e7c6f2.2)]
// CHECK:STDOUT:   %Core.import_ref.db6: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.cd9 = impl_witness_table (%Core.import_ref.db6), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b6: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.3faef3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.44b51c.2)]
// CHECK:STDOUT:   %Core.import_ref.522: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.074 = impl_witness_table (%Core.import_ref.522), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.bcc: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.6ccfae.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2beecc.2)]
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
// CHECK:STDOUT:   %int_1.loc482: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc482: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc482: <bound method> = bound_method %int_1.loc482, %impl.elem0.loc482 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc482: init %Cpp.long_long = call %bound_method.loc482(%int_1.loc482) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc482_3: init %Cpp.long_long = converted %int_1.loc482, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc482 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc482_3
// CHECK:STDOUT:   %.loc482_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc483: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc483: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc483_5.1: %.f7b = impl_witness_access constants.%AddAssignWith.impl_witness.767, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2]
// CHECK:STDOUT:   %bound_method.loc483_5.1: <bound method> = bound_method %a.ref.loc483, %impl.elem0.loc483_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc483: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc483_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc483 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc483: <specific function> = specific_function %impl.elem0.loc483_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.1]
// CHECK:STDOUT:   %bound_method.loc483_5.2: <bound method> = bound_method %a.ref.loc483, %specific_fn.loc483
// CHECK:STDOUT:   %.loc483_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = specific_constant imports.%Core.Op.57a, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1]
// CHECK:STDOUT:   %Op.ref.loc483: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = name_ref Op, %.loc483_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc483, %Op.ref.loc483
// CHECK:STDOUT:   %impl.elem0.loc483_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc483_5.3: <bound method> = bound_method %b.ref.loc483, %impl.elem0.loc483_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc483_5: init %Cpp.long_long = call %bound_method.loc483_5.3(%b.ref.loc483)
// CHECK:STDOUT:   %.loc483_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc483_5
// CHECK:STDOUT:   %.loc483_5.4: %Cpp.long_long = converted %b.ref.loc483, %.loc483_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc483, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.2]
// CHECK:STDOUT:   %bound_method.loc483_5.4: <bound method> = bound_method %a.ref.loc483, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc483_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc483_8: <bound method> = bound_method %b.ref.loc483, %impl.elem0.loc483_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc483_8: init %Cpp.long_long = call %bound_method.loc483_8(%b.ref.loc483)
// CHECK:STDOUT:   %.loc483_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc483_8
// CHECK:STDOUT:   %.loc483_8.2: %Cpp.long_long = converted %b.ref.loc483, %.loc483_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc483_5.4(%a.ref.loc483, %.loc483_8.2)
// CHECK:STDOUT:   %a.ref.loc484: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc484: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc484_5.1: %.b43 = impl_witness_access constants.%SubAssignWith.impl_witness.089, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.2]
// CHECK:STDOUT:   %bound_method.loc484_5.1: <bound method> = bound_method %a.ref.loc484, %impl.elem0.loc484_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc484: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc484_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc484 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc484: <specific function> = specific_function %impl.elem0.loc484_5.1, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.39aa60.1]
// CHECK:STDOUT:   %bound_method.loc484_5.2: <bound method> = bound_method %a.ref.loc484, %specific_fn.loc484
// CHECK:STDOUT:   %.loc484_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.1 = specific_constant imports.%Core.Op.75b, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.1]
// CHECK:STDOUT:   %Op.ref.loc484: %Cpp.long_long.as.SubAssignWith.impl.Op.type.b97f87.1 = name_ref Op, %.loc484_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.268b0a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc484, %Op.ref.loc484
// CHECK:STDOUT:   %impl.elem0.loc484_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc484_5.3: <bound method> = bound_method %b.ref.loc484, %impl.elem0.loc484_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc484_5: init %Cpp.long_long = call %bound_method.loc484_5.3(%b.ref.loc484)
// CHECK:STDOUT:   %.loc484_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc484_5
// CHECK:STDOUT:   %.loc484_5.4: %Cpp.long_long = converted %b.ref.loc484, %.loc484_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc484, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.39aa60.2]
// CHECK:STDOUT:   %bound_method.loc484_5.4: <bound method> = bound_method %a.ref.loc484, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc484_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc484_8: <bound method> = bound_method %b.ref.loc484, %impl.elem0.loc484_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc484_8: init %Cpp.long_long = call %bound_method.loc484_8(%b.ref.loc484)
// CHECK:STDOUT:   %.loc484_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc484_8
// CHECK:STDOUT:   %.loc484_8.2: %Cpp.long_long = converted %b.ref.loc484, %.loc484_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc484_5.4(%a.ref.loc484, %.loc484_8.2)
// CHECK:STDOUT:   %a.ref.loc485: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc485: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc485_5.1: %.ebf = impl_witness_access constants.%MulAssignWith.impl_witness.435, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.2]
// CHECK:STDOUT:   %bound_method.loc485_5.1: <bound method> = bound_method %a.ref.loc485, %impl.elem0.loc485_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc485: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc485_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc485 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc485: <specific function> = specific_function %impl.elem0.loc485_5.1, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.8da25e.1]
// CHECK:STDOUT:   %bound_method.loc485_5.2: <bound method> = bound_method %a.ref.loc485, %specific_fn.loc485
// CHECK:STDOUT:   %.loc485_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.1 = specific_constant imports.%Core.Op.94e, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.1]
// CHECK:STDOUT:   %Op.ref.loc485: %Cpp.long_long.as.MulAssignWith.impl.Op.type.9015f3.1 = name_ref Op, %.loc485_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.e299d7.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc485, %Op.ref.loc485
// CHECK:STDOUT:   %impl.elem0.loc485_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc485_5.3: <bound method> = bound_method %b.ref.loc485, %impl.elem0.loc485_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc485_5: init %Cpp.long_long = call %bound_method.loc485_5.3(%b.ref.loc485)
// CHECK:STDOUT:   %.loc485_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc485_5
// CHECK:STDOUT:   %.loc485_5.4: %Cpp.long_long = converted %b.ref.loc485, %.loc485_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc485, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.8da25e.2]
// CHECK:STDOUT:   %bound_method.loc485_5.4: <bound method> = bound_method %a.ref.loc485, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc485_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc485_8: <bound method> = bound_method %b.ref.loc485, %impl.elem0.loc485_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc485_8: init %Cpp.long_long = call %bound_method.loc485_8(%b.ref.loc485)
// CHECK:STDOUT:   %.loc485_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc485_8
// CHECK:STDOUT:   %.loc485_8.2: %Cpp.long_long = converted %b.ref.loc485, %.loc485_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc485_5.4(%a.ref.loc485, %.loc485_8.2)
// CHECK:STDOUT:   %a.ref.loc486: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc486: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc486_5.1: %.9aa = impl_witness_access constants.%DivAssignWith.impl_witness.817, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.2]
// CHECK:STDOUT:   %bound_method.loc486_5.1: <bound method> = bound_method %a.ref.loc486, %impl.elem0.loc486_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc486: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc486_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc486 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc486: <specific function> = specific_function %impl.elem0.loc486_5.1, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.a29bb8.1]
// CHECK:STDOUT:   %bound_method.loc486_5.2: <bound method> = bound_method %a.ref.loc486, %specific_fn.loc486
// CHECK:STDOUT:   %.loc486_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.1 = specific_constant imports.%Core.Op.ec7, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.1]
// CHECK:STDOUT:   %Op.ref.loc486: %Cpp.long_long.as.DivAssignWith.impl.Op.type.868bc4.1 = name_ref Op, %.loc486_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.6729c8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc486, %Op.ref.loc486
// CHECK:STDOUT:   %impl.elem0.loc486_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc486_5.3: <bound method> = bound_method %b.ref.loc486, %impl.elem0.loc486_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc486_5: init %Cpp.long_long = call %bound_method.loc486_5.3(%b.ref.loc486)
// CHECK:STDOUT:   %.loc486_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc486_5
// CHECK:STDOUT:   %.loc486_5.4: %Cpp.long_long = converted %b.ref.loc486, %.loc486_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc486, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.a29bb8.2]
// CHECK:STDOUT:   %bound_method.loc486_5.4: <bound method> = bound_method %a.ref.loc486, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc486_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc486_8: <bound method> = bound_method %b.ref.loc486, %impl.elem0.loc486_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc486_8: init %Cpp.long_long = call %bound_method.loc486_8(%b.ref.loc486)
// CHECK:STDOUT:   %.loc486_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc486_8
// CHECK:STDOUT:   %.loc486_8.2: %Cpp.long_long = converted %b.ref.loc486, %.loc486_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc486_5.4(%a.ref.loc486, %.loc486_8.2)
// CHECK:STDOUT:   %a.ref.loc487: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc487: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc487_5.1: %.eb2 = impl_witness_access constants.%ModAssignWith.impl_witness.f7c, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.49994b.2]
// CHECK:STDOUT:   %bound_method.loc487_5.1: <bound method> = bound_method %a.ref.loc487, %impl.elem0.loc487_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc487: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc487_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc487 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc487: <specific function> = specific_function %impl.elem0.loc487_5.1, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.e3c8dd.1]
// CHECK:STDOUT:   %bound_method.loc487_5.2: <bound method> = bound_method %a.ref.loc487, %specific_fn.loc487
// CHECK:STDOUT:   %.loc487_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.1 = specific_constant imports.%Core.Op.6fd, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.49994b.1]
// CHECK:STDOUT:   %Op.ref.loc487: %Cpp.long_long.as.ModAssignWith.impl.Op.type.afaa70.1 = name_ref Op, %.loc487_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.49994b.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc487, %Op.ref.loc487
// CHECK:STDOUT:   %impl.elem0.loc487_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc487_5.3: <bound method> = bound_method %b.ref.loc487, %impl.elem0.loc487_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc487_5: init %Cpp.long_long = call %bound_method.loc487_5.3(%b.ref.loc487)
// CHECK:STDOUT:   %.loc487_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc487_5
// CHECK:STDOUT:   %.loc487_5.4: %Cpp.long_long = converted %b.ref.loc487, %.loc487_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc487, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.e3c8dd.2]
// CHECK:STDOUT:   %bound_method.loc487_5.4: <bound method> = bound_method %a.ref.loc487, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc487_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc487_8: <bound method> = bound_method %b.ref.loc487, %impl.elem0.loc487_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc487_8: init %Cpp.long_long = call %bound_method.loc487_8(%b.ref.loc487)
// CHECK:STDOUT:   %.loc487_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc487_8
// CHECK:STDOUT:   %.loc487_8.2: %Cpp.long_long = converted %b.ref.loc487, %.loc487_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc487_5.4(%a.ref.loc487, %.loc487_8.2)
// CHECK:STDOUT:   %a.ref.loc489: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc489: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc489_5.1: %.ba8 = impl_witness_access constants.%BitAndAssignWith.impl_witness.80c, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.2]
// CHECK:STDOUT:   %bound_method.loc489_5.1: <bound method> = bound_method %a.ref.loc489, %impl.elem0.loc489_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc489: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc489_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc489 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc489: <specific function> = specific_function %impl.elem0.loc489_5.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.00c338.1]
// CHECK:STDOUT:   %bound_method.loc489_5.2: <bound method> = bound_method %a.ref.loc489, %specific_fn.loc489
// CHECK:STDOUT:   %.loc489_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.1 = specific_constant imports.%Core.Op.c78, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.1]
// CHECK:STDOUT:   %Op.ref.loc489: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.9d1673.1 = name_ref Op, %.loc489_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.9241d8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc489, %Op.ref.loc489
// CHECK:STDOUT:   %impl.elem0.loc489_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc489_5.3: <bound method> = bound_method %b.ref.loc489, %impl.elem0.loc489_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc489_5: init %Cpp.long_long = call %bound_method.loc489_5.3(%b.ref.loc489)
// CHECK:STDOUT:   %.loc489_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc489_5
// CHECK:STDOUT:   %.loc489_5.4: %Cpp.long_long = converted %b.ref.loc489, %.loc489_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc489, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.00c338.2]
// CHECK:STDOUT:   %bound_method.loc489_5.4: <bound method> = bound_method %a.ref.loc489, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc489_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc489_8: <bound method> = bound_method %b.ref.loc489, %impl.elem0.loc489_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc489_8: init %Cpp.long_long = call %bound_method.loc489_8(%b.ref.loc489)
// CHECK:STDOUT:   %.loc489_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc489_8
// CHECK:STDOUT:   %.loc489_8.2: %Cpp.long_long = converted %b.ref.loc489, %.loc489_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc489_5.4(%a.ref.loc489, %.loc489_8.2)
// CHECK:STDOUT:   %a.ref.loc490: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc490: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc490_5.1: %.4fb = impl_witness_access constants.%BitOrAssignWith.impl_witness.bd2, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.2]
// CHECK:STDOUT:   %bound_method.loc490_5.1: <bound method> = bound_method %a.ref.loc490, %impl.elem0.loc490_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc490: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc490_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc490 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc490: <specific function> = specific_function %impl.elem0.loc490_5.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.b0c3cb.1]
// CHECK:STDOUT:   %bound_method.loc490_5.2: <bound method> = bound_method %a.ref.loc490, %specific_fn.loc490
// CHECK:STDOUT:   %.loc490_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.1 = specific_constant imports.%Core.Op.fa0, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.1]
// CHECK:STDOUT:   %Op.ref.loc490: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.4196b6.1 = name_ref Op, %.loc490_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.acdb6d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc490, %Op.ref.loc490
// CHECK:STDOUT:   %impl.elem0.loc490_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc490_5.3: <bound method> = bound_method %b.ref.loc490, %impl.elem0.loc490_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc490_5: init %Cpp.long_long = call %bound_method.loc490_5.3(%b.ref.loc490)
// CHECK:STDOUT:   %.loc490_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc490_5
// CHECK:STDOUT:   %.loc490_5.4: %Cpp.long_long = converted %b.ref.loc490, %.loc490_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc490, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.b0c3cb.2]
// CHECK:STDOUT:   %bound_method.loc490_5.4: <bound method> = bound_method %a.ref.loc490, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc490_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc490_8: <bound method> = bound_method %b.ref.loc490, %impl.elem0.loc490_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc490_8: init %Cpp.long_long = call %bound_method.loc490_8(%b.ref.loc490)
// CHECK:STDOUT:   %.loc490_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc490_8
// CHECK:STDOUT:   %.loc490_8.2: %Cpp.long_long = converted %b.ref.loc490, %.loc490_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc490_5.4(%a.ref.loc490, %.loc490_8.2)
// CHECK:STDOUT:   %a.ref.loc491: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc491: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc491_5.1: %.919 = impl_witness_access constants.%BitXorAssignWith.impl_witness.f91, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.2]
// CHECK:STDOUT:   %bound_method.loc491_5.1: <bound method> = bound_method %a.ref.loc491, %impl.elem0.loc491_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc491: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc491_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc491 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc491: <specific function> = specific_function %impl.elem0.loc491_5.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8f3086.1]
// CHECK:STDOUT:   %bound_method.loc491_5.2: <bound method> = bound_method %a.ref.loc491, %specific_fn.loc491
// CHECK:STDOUT:   %.loc491_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.1 = specific_constant imports.%Core.Op.3d4, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.1]
// CHECK:STDOUT:   %Op.ref.loc491: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.7f7009.1 = name_ref Op, %.loc491_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3aa346.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc491, %Op.ref.loc491
// CHECK:STDOUT:   %impl.elem0.loc491_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc491_5.3: <bound method> = bound_method %b.ref.loc491, %impl.elem0.loc491_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc491_5: init %Cpp.long_long = call %bound_method.loc491_5.3(%b.ref.loc491)
// CHECK:STDOUT:   %.loc491_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc491_5
// CHECK:STDOUT:   %.loc491_5.4: %Cpp.long_long = converted %b.ref.loc491, %.loc491_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc491, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.8f3086.2]
// CHECK:STDOUT:   %bound_method.loc491_5.4: <bound method> = bound_method %a.ref.loc491, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc491_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc491_8: <bound method> = bound_method %b.ref.loc491, %impl.elem0.loc491_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc491_8: init %Cpp.long_long = call %bound_method.loc491_8(%b.ref.loc491)
// CHECK:STDOUT:   %.loc491_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc491_8
// CHECK:STDOUT:   %.loc491_8.2: %Cpp.long_long = converted %b.ref.loc491, %.loc491_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc491_5.4(%a.ref.loc491, %.loc491_8.2)
// CHECK:STDOUT:   %a.ref.loc492: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc492: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc492_5.1: %.64f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.c81, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.2]
// CHECK:STDOUT:   %bound_method.loc492_5.1: <bound method> = bound_method %a.ref.loc492, %impl.elem0.loc492_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc492: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc492_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc492 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc492: <specific function> = specific_function %impl.elem0.loc492_5.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1c7fb8.1]
// CHECK:STDOUT:   %bound_method.loc492_5.2: <bound method> = bound_method %a.ref.loc492, %specific_fn.loc492
// CHECK:STDOUT:   %.loc492_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.1 = specific_constant imports.%Core.Op.1b6, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.1]
// CHECK:STDOUT:   %Op.ref.loc492: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.21cdd1.1 = name_ref Op, %.loc492_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.e91293.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc492, %Op.ref.loc492
// CHECK:STDOUT:   %impl.elem0.loc492_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc492_5.3: <bound method> = bound_method %b.ref.loc492, %impl.elem0.loc492_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc492_5: init %Cpp.long_long = call %bound_method.loc492_5.3(%b.ref.loc492)
// CHECK:STDOUT:   %.loc492_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc492_5
// CHECK:STDOUT:   %.loc492_5.4: %Cpp.long_long = converted %b.ref.loc492, %.loc492_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc492, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.1c7fb8.2]
// CHECK:STDOUT:   %bound_method.loc492_5.4: <bound method> = bound_method %a.ref.loc492, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc492_9: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc492_9: <bound method> = bound_method %b.ref.loc492, %impl.elem0.loc492_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc492_9: init %Cpp.long_long = call %bound_method.loc492_9(%b.ref.loc492)
// CHECK:STDOUT:   %.loc492_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc492_9
// CHECK:STDOUT:   %.loc492_9.2: %Cpp.long_long = converted %b.ref.loc492, %.loc492_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc492_5.4(%a.ref.loc492, %.loc492_9.2)
// CHECK:STDOUT:   %a.ref.loc493: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc493: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc493_5.1: %.298 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.1b7, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.2]
// CHECK:STDOUT:   %bound_method.loc493_5.1: <bound method> = bound_method %a.ref.loc493, %impl.elem0.loc493_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc493: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc493_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc493 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc493: <specific function> = specific_function %impl.elem0.loc493_5.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1c2239.1]
// CHECK:STDOUT:   %bound_method.loc493_5.2: <bound method> = bound_method %a.ref.loc493, %specific_fn.loc493
// CHECK:STDOUT:   %.loc493_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.1 = specific_constant imports.%Core.Op.bcc, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.1]
// CHECK:STDOUT:   %Op.ref.loc493: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.0ddf2d.1 = name_ref Op, %.loc493_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.efc522.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc493, %Op.ref.loc493
// CHECK:STDOUT:   %impl.elem0.loc493_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc493_5.3: <bound method> = bound_method %b.ref.loc493, %impl.elem0.loc493_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc493_5: init %Cpp.long_long = call %bound_method.loc493_5.3(%b.ref.loc493)
// CHECK:STDOUT:   %.loc493_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc493_5
// CHECK:STDOUT:   %.loc493_5.4: %Cpp.long_long = converted %b.ref.loc493, %.loc493_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc493, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1c2239.2]
// CHECK:STDOUT:   %bound_method.loc493_5.4: <bound method> = bound_method %a.ref.loc493, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc493_9: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc493_9: <bound method> = bound_method %b.ref.loc493, %impl.elem0.loc493_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc493_9: init %Cpp.long_long = call %bound_method.loc493_9(%b.ref.loc493)
// CHECK:STDOUT:   %.loc493_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc493_9
// CHECK:STDOUT:   %.loc493_9.2: %Cpp.long_long = converted %b.ref.loc493, %.loc493_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc493_5.4(%a.ref.loc493, %.loc493_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.ea5: %ImplicitAs.type.a03 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.e3f: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.815, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.d69079.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.d69079.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.e3f) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.58a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.58a, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.918060.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.d69079.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.918060.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.d69079.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.277: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.815 = impl_witness_table (%Core.import_ref.277), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.57a: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.76e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc505: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc505: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc505: <bound method> = bound_method %int_1.loc505, %impl.elem0.loc505 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc505: init %Cpp.long_long = call %bound_method.loc505(%int_1.loc505) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc505_3: init %Cpp.long_long = converted %int_1.loc505, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc505 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc505_3
// CHECK:STDOUT:   %.loc505_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc506: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc506_5.1: %.883 = impl_witness_access constants.%AddAssignWith.impl_witness.e3f, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d69079.2]
// CHECK:STDOUT:   %bound_method.loc506_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc506_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc506_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc506_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.918060.1]
// CHECK:STDOUT:   %bound_method.loc506_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc506_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.1 = specific_constant imports.%Core.Op.57a, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d69079.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.11fd63.1 = name_ref Op, %.loc506_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.d69079.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc506_5.2: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc506_5.3: <bound method> = bound_method %int_1.loc506, %impl.elem0.loc506_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc506_5: init %Cpp.long_long = call %bound_method.loc506_5.3(%int_1.loc506) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc506_5.3: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc506_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc506_5.4: %Cpp.long_long = converted %int_1.loc506, %.loc506_5.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.918060.2]
// CHECK:STDOUT:   %bound_method.loc506_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc506_8: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc506_8: <bound method> = bound_method %int_1.loc506, %impl.elem0.loc506_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc506_8: init %Cpp.long_long = call %bound_method.loc506_8(%int_1.loc506) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc506_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc506_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc506_8.2: %Cpp.long_long = converted %int_1.loc506, %.loc506_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc506_5.4(%a.ref, %.loc506_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_runtime_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
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
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.ea5) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1b3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b19 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c59: %ImplicitAs.type.a03 = facet_value %i64, (%ImplicitAs.impl_witness.1b3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.767: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.815, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.d84 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.767) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d84: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.f7b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d84, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.653: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.653, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.277: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.815 = impl_witness_table (%Core.import_ref.277), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.57a: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.3e35db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf74e0.2)]
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
// CHECK:STDOUT:   %int_1.loc518: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc518: %.a119 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc518: <bound method> = bound_method %int_1.loc518, %impl.elem0.loc518 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc518: init %Cpp.long_long = call %bound_method.loc518(%int_1.loc518) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc518_3: init %Cpp.long_long = converted %int_1.loc518, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc518 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc518_3
// CHECK:STDOUT:   %.loc518_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc519: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc519: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc519_16.1: <bound method> = bound_method %int_1.loc519, %impl.elem0.loc519 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc519: <specific function> = specific_function %impl.elem0.loc519, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc519_16.2: <bound method> = bound_method %int_1.loc519, %specific_fn.loc519 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc519: init %i64 = call %bound_method.loc519_16.2(%int_1.loc519) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc519_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc519 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc519_16.2: %i64 = converted %int_1.loc519, %.loc519_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc519_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc520_5.1: %.f7b = impl_witness_access constants.%AddAssignWith.impl_witness.767, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.2]
// CHECK:STDOUT:   %bound_method.loc520_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc520_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc520_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc520: <specific function> = specific_function %impl.elem0.loc520_5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.1]
// CHECK:STDOUT:   %bound_method.loc520_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc520
// CHECK:STDOUT:   %.loc520_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = specific_constant imports.%Core.Op.57a, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.2293dc.1 = name_ref Op, %.loc520_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.fa1912.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc520_5.2: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc520_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc520_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc520_5: init %Cpp.long_long = call %bound_method.loc520_5.3(%b.ref)
// CHECK:STDOUT:   %.loc520_5.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc520_5
// CHECK:STDOUT:   %.loc520_5.4: %Cpp.long_long = converted %b.ref, %.loc520_5.3
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.50a783.2]
// CHECK:STDOUT:   %bound_method.loc520_5.4: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc520_8: %.cb1 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc520_8: <bound method> = bound_method %b.ref, %impl.elem0.loc520_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc520_8: init %Cpp.long_long = call %bound_method.loc520_8(%b.ref)
// CHECK:STDOUT:   %.loc520_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc520_8
// CHECK:STDOUT:   %.loc520_8.2: %Cpp.long_long = converted %b.ref, %.loc520_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc520_5.4(%a.ref, %.loc520_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc562: %.a11 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc562: <bound method> = bound_method %int_1, %impl.elem0.loc562 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc562(%int_1) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc562_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   assign %a.var, %.loc562_3
// CHECK:STDOUT:   %.loc562_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc563: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc563: %.11e = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc563: <bound method> = bound_method %a.ref.loc563, %impl.elem0.loc563
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc563(%a.ref.loc563)
// CHECK:STDOUT:   %a.ref.loc564: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc564: %.c6e = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc564: <bound method> = bound_method %a.ref.loc564, %impl.elem0.loc564
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc564(%a.ref.loc564)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long_long) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4384: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.195: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.4384) [concrete]
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
// CHECK:STDOUT:   %int_1.loc583: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc583_34: type = splice_block %unsigned_long_long.ref.loc583 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc583: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc583: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc583: %.30f = impl_witness_access constants.%ImplicitAs.impl_witness.4384, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f87]
// CHECK:STDOUT:   %bound_method.loc583: <bound method> = bound_method %int_1.loc583, %impl.elem0.loc583 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc583: init %Cpp.unsigned_long_long = call %bound_method.loc583(%int_1.loc583) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc583_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc583 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc583_56.2: %Cpp.unsigned_long_long = converted %int_1.loc583, %.loc583_56.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = value_binding cpp_unsigned_long_long, %.loc583_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.633 = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc584_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc584: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc584: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc584_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc584: %ptr.d19 = addr_of %.loc584_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc584: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_unsigned_long_long.ref.loc584, %addr.loc584)
// CHECK:STDOUT:   %.loc584_107.2: init %ULongLongResult to %.loc584_107.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc584
// CHECK:STDOUT:   %.loc584_48: type = splice_block %ULongLongResult.ref.loc584 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc584_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc584: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc584_107.3: ref %ULongLongResult = temporary %.loc584_107.1, %.loc584_107.2
// CHECK:STDOUT:   %.loc584_107.4: %ULongLongResult = acquire_value %.loc584_107.3
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = value_binding cpp_unsigned_long_long_result, %.loc584_107.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long.patt: %pattern_type.ebd = value_binding_pattern cpp_compat_ulong_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc586: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc586: type = splice_block %ULongLong64.ref [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %ULongLong64.ref: type = name_ref ULongLong64, imports.%Core.ULongLong64 [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long: %Cpp.unsigned_long_long = value_binding cpp_compat_ulong_long, %cpp_unsigned_long_long.ref.loc586
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long_result.patt: %pattern_type.633 = value_binding_pattern cpp_compat_ulong_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc587_66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc587: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc587_105.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc587: %ptr.d19 = addr_of %.loc587_105.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc587: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_compat_ulong_long.ref, %addr.loc587)
// CHECK:STDOUT:   %.loc587_105.2: init %ULongLongResult to %.loc587_105.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc587
// CHECK:STDOUT:   %.loc587_47: type = splice_block %ULongLongResult.ref.loc587 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc587_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc587: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc587_105.3: ref %ULongLongResult = temporary %.loc587_105.1, %.loc587_105.2
// CHECK:STDOUT:   %.loc587_105.4: %ULongLongResult = acquire_value %.loc587_105.3
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = value_binding cpp_compat_ulong_long_result, %.loc587_105.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.157 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc589: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc589: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc589_25.1: <bound method> = bound_method %int_1.loc589, %impl.elem0.loc589 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.141]
// CHECK:STDOUT:   %specific_fn.loc589: <specific function> = specific_function %impl.elem0.loc589, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc589_25.2: <bound method> = bound_method %int_1.loc589, %specific_fn.loc589 [concrete = constants.%bound_method.fec]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc589: init %u64 = call %bound_method.loc589_25.2(%int_1.loc589) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc589_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc589 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc589_25.2: %u64 = converted %int_1.loc589, %.loc589_25.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %carbon_u64: %u64 = value_binding carbon_u64, %.loc589_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.6f2 = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc590_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc590: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc590_79.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc590: %ptr.f5e = addr_of %.loc590_79.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc590: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.2(%carbon_u64.ref, %addr.loc590)
// CHECK:STDOUT:   %.loc590_79.2: init %ULongResult to %.loc590_79.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc590
// CHECK:STDOUT:   %.loc590_36: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc590_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc590_79.3: ref %ULongResult = temporary %.loc590_79.1, %.loc590_79.2
// CHECK:STDOUT:   %.loc590_79.4: %ULongResult = acquire_value %.loc590_79.3
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = value_binding carbon_u64_result, %.loc590_79.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc592_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc592_64: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc592_64.1: <bound method> = bound_method %float, %impl.elem0.loc592_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc592: <specific function> = specific_function %impl.elem0.loc592_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc592_64.2: <bound method> = bound_method %float, %specific_fn.loc592 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc592_64.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc592_64.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc592_64.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc592_64.4: init %f32.97e to %.loc592_64.3 = in_place_init %.loc592_64.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc592_64.5: init %array_type to %a.var = array_init (%.loc592_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc592_3: init %array_type = converted %.loc592_64.1, %.loc592_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc592_3
// CHECK:STDOUT:   %.loc592_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc592: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc592: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc592: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc592_30.1: %.83e = impl_witness_access constants.%As.impl_witness.d1f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc592_30.1: <bound method> = bound_method %int_1.loc592, %impl.elem0.loc592_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc592_30.1(%int_1.loc592) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc592_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc592_30.2: %Cpp.unsigned_long_long = converted %int_1.loc592, %.loc592_30.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %impl.elem0.loc592_30.2: %.66f = impl_witness_access constants.%ImplicitAs.impl_witness.7c3, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc592_30.2: <bound method> = bound_method %.loc592_30.2, %impl.elem0.loc592_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc592_30.2(%.loc592_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc592_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc592_30.4: Core.IntLiteral = converted %.loc592_30.2, %.loc592_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc592_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound: <bound method> = bound_method %.loc590_79.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound(%.loc590_79.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc587: <bound method> = bound_method %.loc587_105.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc587: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc587(%.loc587_105.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc584: <bound method> = bound_method %.loc584_107.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc584: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc584(%.loc584_107.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.407: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4384: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.4384) [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc629: %.30f = impl_witness_access constants.%ImplicitAs.impl_witness.4384, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc629: <bound method> = bound_method %int_1, %impl.elem0.loc629 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc629(%int_1) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc629_3: init %Cpp.unsigned_long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   assign %a.var, %.loc629_3
// CHECK:STDOUT:   %.loc629_13: type = splice_block %unsigned_long_long.ref.loc629 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc629: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc629: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ebd = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ebd = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc630_42: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc630: %.d24 = impl_witness_access constants.%Copy.impl_witness.95c, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc630: <bound method> = bound_method %.loc630_42, %impl.elem0.loc630
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc630(%.loc630_42)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc630_20: type = splice_block %unsigned_long_long.ref.loc630 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc630: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc630: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc630: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc630: init %empty_tuple.type = call %Destroy.Op.bound.loc630(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc629: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc629: init %empty_tuple.type = call %Destroy.Op.bound.loc629(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.unsigned_long_long) = "no_op";
// CHECK:STDOUT:
