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
  // CHECK:STDERR: fail_implicit_conversion_long_long_to_i64.carbon:[[@LINE+7]]:32: error: cannot implicitly convert expression of type `long_long` to `i64` [ConversionFailure]
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
  // CHECK:STDERR: fail_pass_ulonglong_to_u64.carbon:[[@LINE+11]]:25: error: cannot implicitly convert expression of type `unsigned_long_long` to `u64` [ConversionFailure]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %cpp_long.patt: %pattern_type.6d9 = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.122: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5f8: %ImplicitAs.type.270 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.122) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.de7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.5f8) [concrete]
// CHECK:STDOUT:   %.19a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.de7, %ImplicitAs.facet.5f8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.013: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.4f6: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_long.patt: %pattern_type.6d9 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.20c: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.291: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.051: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.20c: %Core.IntLiteral.as.As.impl.Convert.type.051 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.20c = facet_value Core.IntLiteral, (%As.impl_witness.291) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.96d: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.e74: type = fn_type_with_self_type %As.WithSelf.Convert.type.96d, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.20c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.20c, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.29d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8f7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.e14: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.33b: %Int.as.ImplicitAs.impl.Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3b5: %ImplicitAs.type.5be = facet_value %i64, (%ImplicitAs.impl_witness.8f7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.98c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.3b5) [concrete]
// CHECK:STDOUT:   %.8f5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.98c, %ImplicitAs.facet.3b5 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.4f6, %Int.as.ImplicitAs.impl.Convert.33b [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.33b, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.09e: <bound method> = bound_method %int_1.4f6, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.984: type = class_type @Array, @Array(%f32.2d9, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.32f: type = pattern_type %Array.984 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0f1, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec7: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet.ec7) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet.ec7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.29b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.984 = tuple_value (%float.3a8) [concrete]
// CHECK:STDOUT:   %self.param_patt.ab7: %pattern_type.afb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20c: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt.ab7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3dc: %pattern_type.32f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e84: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt.3dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc11_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc11_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.19a = impl_witness_access constants.%ImplicitAs.impl_witness.122, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.64f]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.013]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_28.2(%int_1.loc8) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc8_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc8_28.2: %i64 = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long: %i64 = wrapper_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.6d9 = value_binding_pattern cpp_long [concrete = constants.%cpp_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = wrapper_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.6d9 = value_binding_pattern carbon_long [concrete = constants.%carbon_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_50: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn.loc11_50 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.2d9 = call %bound_method.loc11_50.2(%float) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc11_50.2: init %f32.2d9 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %f32.2d9 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %f32.2d9 to %.loc11_50.3 = in_place_init %.loc11_50.2 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc11_50.5: init %Array.984 to %a.var = array_init (%.loc11_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %Array.984 = converted %.loc11_50.1, %.loc11_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_41: type = splice_block %Array [concrete = constants.%Array.984] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.e74 = impl_witness_access constants.%As.impl_witness.291, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.20c]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.29d]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.8f5 = impl_witness_access constants.%ImplicitAs.impl_witness.8f7, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.33b]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.09e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.2d9, constants.%int_1.5b8) [concrete = constants.%Array.984]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.984 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.1(%self.param: ref %f32.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11: ref %f32.2d9 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.2(%self.param: ref %Array.984) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.878 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.615: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.7db(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.edd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.240: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.020(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.649: %UInt.as.ImplicitAs.impl.Convert.type.240 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bf2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0ef: %ImplicitAs.type.615 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a86) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b73: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.0ef) [concrete]
// CHECK:STDOUT:   %.072: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b73, %ImplicitAs.facet.0ef [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4bb: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d00: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_long.patt: %pattern_type.878 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.8e3: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c31: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.2fa(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.750: %Core.IntLiteral.as.As.impl.Convert.type.c31 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.402: <witness> = impl_witness imports.%As.impl_witness_table.a6a, @Core.IntLiteral.as.As.impl.2fa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.d5c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.2fa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.7e5: %Core.IntLiteral.as.As.impl.Convert.type.d5c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.8e3 = facet_value Core.IntLiteral, (%As.impl_witness.402) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.5f4: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.1ea1: type = fn_type_with_self_type %As.WithSelf.Convert.type.5f4, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.7e5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.7e5, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.1fc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8dc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f1b, @UInt.as.ImplicitAs.impl.020(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.991: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.020(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.fbf: %UInt.as.ImplicitAs.impl.Convert.type.991 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1f6: %ImplicitAs.type.5be = facet_value %u64, (%ImplicitAs.impl_witness.8dc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.630: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.1f6) [concrete]
// CHECK:STDOUT:   %.f64: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.630, %ImplicitAs.facet.1f6 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.d00, %UInt.as.ImplicitAs.impl.Convert.fbf [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.fbf, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d19: <bound method> = bound_method %int_1.d00, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.984: type = class_type @Array, @Array(%f32.2d9, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.32f: type = pattern_type %Array.984 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0f1, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec7: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet.ec7) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet.ec7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.29b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.984 = tuple_value (%float.3a8) [concrete]
// CHECK:STDOUT:   %self.param_patt.ab7: %pattern_type.afb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20c: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt.ab7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3dc: %pattern_type.32f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e84: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt.3dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.602: @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.edd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bf2 = impl_witness_table (%Core.import_ref.602), @Core.IntLiteral.as.ImplicitAs.impl.7db [concrete]
// CHECK:STDOUT:   %Core.import_ref.75f: @UInt.as.ImplicitAs.impl.020.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.240) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.020.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.649)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f1b = impl_witness_table (%Core.import_ref.75f), @UInt.as.ImplicitAs.impl.020 [concrete]
// CHECK:STDOUT:   %Core.import_ref.74f: @Core.IntLiteral.as.As.impl.2fa.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c31) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.2fa.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.750)]
// CHECK:STDOUT:   %As.impl_witness_table.a6a = impl_witness_table (%Core.import_ref.74f), @Core.IntLiteral.as.As.impl.2fa [concrete]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc11_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc11_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.072 = impl_witness_access constants.%ImplicitAs.impl_witness.a86, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.4bb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_46.2(%int_1.loc8) [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc8_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc8_46.2: %u64 = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = wrapper_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.878 = value_binding_pattern cpp_unsigned_long [concrete = constants.%cpp_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = wrapper_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.878 = value_binding_pattern carbon_unsigned_long [concrete = constants.%carbon_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_59: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn.loc11_59 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.2d9 = call %bound_method.loc11_59.2(%float) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc11_59.2: init %f32.2d9 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_59.3: ref %f32.2d9 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_59.4: init %f32.2d9 to %.loc11_59.3 = in_place_init %.loc11_59.2 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc11_59.5: init %Array.984 to %a.var = array_init (%.loc11_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %Array.984 = converted %.loc11_59.1, %.loc11_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_50: type = splice_block %Array [concrete = constants.%Array.984] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.1ea1 = impl_witness_access constants.%As.impl_witness.402, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.7e5]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.1fc]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.d00]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.d00]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.f64 = impl_witness_access constants.%ImplicitAs.impl_witness.8dc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.fbf]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.d19]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.2d9, constants.%int_1.5b8) [concrete = constants.%Array.984]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.984 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.1(%self.param: ref %f32.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11: ref %f32.2d9 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.2(%self.param: ref %Array.984) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %cpp_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
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
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %carbon_i64.patt: %pattern_type.6d9 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %carbon_i64_result.patt: %pattern_type.ae7 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   %ptr.aec: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.d1b: type = facet_type <@As, @As(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.420: <witness> = impl_witness imports.%As.impl_witness_table.81c [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d1b = facet_value Core.IntLiteral, (%As.impl_witness.420) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.339: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.3bf: type = fn_type_with_self_type %As.WithSelf.Convert.type.339, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.392: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.700 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfe: %ImplicitAs.type.5be = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.392) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dc5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.dfe) [concrete]
// CHECK:STDOUT:   %.cbc: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dc5, %ImplicitAs.facet.dfe [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.b19, %Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.984: type = class_type @Array, @Array(%f32.2d9, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.32f: type = pattern_type %Array.984 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0f1, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec7: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet.ec7) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet.ec7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.984 = tuple_value (%float.3a8) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.ab7: %pattern_type.afb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20c: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt.ab7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3dc: %pattern_type.32f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e84: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt.3dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b62: %pattern_type.ae7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5c45: %pattern_type.ae7 = wrapper_binding_pattern self, %self.param_patt.b62 [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.98a: %pattern_type.357 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.784: %pattern_type.357 = wrapper_binding_pattern self, %self.param_patt.98a [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Array = %Core.Array
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
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
// CHECK:STDOUT:   %As.impl_witness_table.81c = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.a6f [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.700 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.long_long.as.ImplicitAs.impl.650 [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/types/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %LongResult.cpp_destructor.decl: %LongResult.cpp_destructor.type = fn_decl @LongResult.cpp_destructor [concrete = constants.%LongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.decl: %LongLongResult.cpp_destructor.type = fn_decl @LongLongResult.cpp_destructor [concrete = constants.%LongLongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc24_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ae7 = ref_param_pattern [concrete = constants.%self.param_patt.b62]
// CHECK:STDOUT:     %self.patt: %pattern_type.ae7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5c45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %LongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %LongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc19 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.357 = ref_param_pattern [concrete = constants.%self.param_patt.98a]
// CHECK:STDOUT:     %self.patt: %pattern_type.357 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.784]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %LongLongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %LongLongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.6d9 = value_binding_pattern carbon_i64 [concrete = constants.%carbon_i64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.2d9 = call %bound_method.loc24_55.2(%float) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_55.2: init %f32.2d9 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_55.3: ref %f32.2d9 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_55.4: init %f32.2d9 to %.loc24_55.3 = in_place_init %.loc24_55.2 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_55.5: init %Array.984 to %a.var = array_init (%.loc24_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %Array.984 = converted %.loc24_55.1, %.loc24_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_46: type = splice_block %Array [concrete = constants.%Array.984] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.3bf = impl_witness_access constants.%As.impl_witness.420, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.cbc = impl_witness_access constants.%ImplicitAs.impl_witness.392, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.2d9, constants.%int_1.5b8) [concrete = constants.%Array.984]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.984 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%a.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_77.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_77.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_101.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_101.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_87.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16(%.loc16_87.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.1(%self.param: ref %f32.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24: ref %f32.2d9 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.2(%self.param: ref %Array.984) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22(%self.param: ref %LongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22(%self.param: ref %LongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%LongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc19(%self.param: ref %LongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc19(%self.param: ref %LongLongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%LongLongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1ae: <witness> = impl_witness imports.%Copy.impl_witness_table.d10 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.1ae) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.449: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.41f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.449, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d10 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.41f = impl_witness_access constants.%Copy.impl_witness.1ae, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %EqWith.type.e35: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f94: <witness> = impl_witness imports.%EqWith.impl_witness_table.d15 [concrete]
// CHECK:STDOUT:   %EqWith.facet.78b: %EqWith.type.e35 = facet_value %Cpp.long_long, (%EqWith.impl_witness.f94) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.dd8: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.78b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.175: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.78b) [concrete]
// CHECK:STDOUT:   %.47c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.dd8, %EqWith.facet.78b [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1ca: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.175, %EqWith.facet.78b [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.a6a: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b28: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.9d7: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.2b7 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.3c8: %OrderedWith.type.b28 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.9d7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ed0: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.3c8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.ba2: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.3c8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.11f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.3c8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.5e9: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.3c8) [concrete]
// CHECK:STDOUT:   %.8a5: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.11f, %OrderedWith.facet.3c8 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.7c8: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.ad5: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ed0, %OrderedWith.facet.3c8 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.606: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.6e8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.5e9, %OrderedWith.facet.3c8 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.923: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.ba2, %OrderedWith.facet.3c8 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a874: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.d15 = impl_witness_table (%Core.import_ref.a874, %Core.import_ref.adc), @Cpp.long_long.as.EqWith.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e88: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.2b7 = impl_witness_table (%Core.import_ref.0e88, %Core.import_ref.757, %Core.import_ref.c32, %Core.import_ref.381), @Cpp.long_long.as.OrderedWith.impl.fb1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.47c = impl_witness_access constants.%EqWith.impl_witness.f94, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.9ed = impl_witness_access constants.%EqWith.impl_witness.f94, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.8a5 = impl_witness_access constants.%OrderedWith.impl_witness.9d7, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.ad5 = impl_witness_access constants.%OrderedWith.impl_witness.9d7, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.6e8 = impl_witness_access constants.%OrderedWith.impl_witness.9d7, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.923 = impl_witness_access constants.%OrderedWith.impl_witness.9d7, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.3c0: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.d70: %ImplicitAs.type.a57 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.e41(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.c34842.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.e41(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.8d6670.1: %Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.e41(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.c34842.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.e41(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.8d6670.2: %Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.61f: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4b2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9d1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecb: %ImplicitAs.type.a57 = facet_value %i64, (%ImplicitAs.impl_witness.4b2) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.c33: <witness> = impl_witness imports.%EqWith.impl_witness_table.786, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a07732.1: %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.679372.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a07732.2: %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.679372.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.20d: %EqWith.type.3c0 = facet_value %Cpp.long_long, (%EqWith.impl_witness.c33) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.255: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.20d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0ee: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.20d) [concrete]
// CHECK:STDOUT:   %.ab5: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.255, %EqWith.facet.20d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.b1ebc1.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a07732.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.b1ebc1.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a07732.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.96a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.96a, %ImplicitAs.facet.ecb [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ac5: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0ee, %EqWith.facet.20d [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.7cb166.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.679372.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.7cb166.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.679372.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.914: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.de377b.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.085a19.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.e15d49.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.366178.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.ede047.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.366178.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.de377b.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.085a19.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.e15d49.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.366178.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.f88(%T.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.ede047.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.366178.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.024: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.d19: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.163, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.e75d27.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.e75d27.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.a40: %OrderedWith.type.914 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.d19) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.598: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.a40) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.4f1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.a40) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.730: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.a40) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ef4: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.a40) [concrete]
// CHECK:STDOUT:   %.603: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.730, %OrderedWith.facet.a40 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.178f95.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.178f95.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.358a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.598, %OrderedWith.facet.a40 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.dbdfd4.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.e75d27.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.dbdfd4.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.e75d27.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.202: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ef4, %OrderedWith.facet.a40 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.027f71.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.027f71.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.ead: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.4f1, %OrderedWith.facet.a40 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.783ffb.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.783ffb.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.abe: <witness> = impl_witness imports.%EqWith.impl_witness_table.786, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a699b9.1: %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.870d75.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.a699b9.2: %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.e41(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.870d75.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.0a0: %EqWith.type.61f = facet_value %Cpp.long_long, (%EqWith.impl_witness.abe) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.ee7: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.0a0) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.806: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.0a0) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.ee7, %EqWith.facet.0a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.78fda2.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a699b9.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.78fda2.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.a699b9.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.b96: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.806, %EqWith.facet.0a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.6e28cf.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.870d75.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.6e28cf.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.870d75.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.184: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.163, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.963b6f.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.963b6f.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.f88(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.91e: %OrderedWith.type.024 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.184) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.457: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.91e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.f2e: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.91e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.021: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.91e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.da1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.91e) [concrete]
// CHECK:STDOUT:   %.428: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.021, %OrderedWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.ac49c5.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.ac49c5.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.18d: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.457, %OrderedWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.cca121.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.963b6f.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.cca121.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.963b6f.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.e5e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.da1, %OrderedWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4ba090.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4ba090.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.90c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.f2e, %OrderedWith.facet.91e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ccb3d.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ccb3d.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.730a: @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.8d6670.1)]
// CHECK:STDOUT:   %Core.import_ref.078d: @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.c34842.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.786 = impl_witness_table (%Core.import_ref.730a, %Core.import_ref.078d), @Cpp.long_long.as.EqWith.impl.e41 [concrete]
// CHECK:STDOUT:   %Core.Equal.22f: @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.ceec12.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.8d6670.2)]
// CHECK:STDOUT:   %Core.NotEqual.861: @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.e703cb.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.e41.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.c34842.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9d1 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.54c: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.366178.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.ede047.1)]
// CHECK:STDOUT:   %Core.import_ref.5ed0: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.e15d49.1)]
// CHECK:STDOUT:   %Core.import_ref.5ea: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.085a19.1)]
// CHECK:STDOUT:   %Core.import_ref.1e9: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.de377b.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.163 = impl_witness_table (%Core.import_ref.54c, %Core.import_ref.5ed0, %Core.import_ref.5ea, %Core.import_ref.1e9), @Cpp.long_long.as.OrderedWith.impl.f88 [concrete]
// CHECK:STDOUT:   %Core.Less.4f0: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.366178.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.ede047.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.602: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.239c14.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.e15d49.2)]
// CHECK:STDOUT:   %Core.Greater.1b9: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9426c1.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.085a19.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.093: @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.f99334.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.f88.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.de377b.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.ab5 = impl_witness_access constants.%EqWith.impl_witness.c33, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a07732.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.b1ebc1.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.1 = specific_constant imports.%Core.Equal.22f, @Cpp.long_long.as.EqWith.impl.e41(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a07732.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.90f0f8.1 = name_ref Equal, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a07732.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.b1ebc1.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.ac5 = impl_witness_access constants.%EqWith.impl_witness.c33, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.679372.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.7cb166.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.1 = specific_constant imports.%Core.NotEqual.861, @Cpp.long_long.as.EqWith.impl.e41(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.679372.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4641e8.1 = name_ref NotEqual, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.679372.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.7cb166.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.603 = impl_witness_access constants.%OrderedWith.impl_witness.d19, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.178f95.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.1 = specific_constant imports.%Core.Greater.1b9, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.d99f00.1 = name_ref Greater, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.ed3a43.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.178f95.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.358a = impl_witness_access constants.%OrderedWith.impl_witness.d19, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.e75d27.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.dbdfd4.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.1 = specific_constant imports.%Core.Less.4f0, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.e75d27.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.6c110d.1 = name_ref Less, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.e75d27.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.dbdfd4.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.202 = impl_witness_access constants.%OrderedWith.impl_witness.d19, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.027f71.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.1 = specific_constant imports.%Core.GreaterOrEquivalent.093, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.c2e054.1 = name_ref GreaterOrEquivalent, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.0071e4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.027f71.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.ead = impl_witness_access constants.%OrderedWith.impl_witness.d19, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.783ffb.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.1 = specific_constant imports.%Core.LessOrEquivalent.602, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.135fc5.1 = name_ref LessOrEquivalent, %.loc15_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.d33e93.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.783ffb.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.070 = impl_witness_access constants.%EqWith.impl_witness.abe, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a699b9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.78fda2.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.1 = specific_constant imports.%Core.Equal.22f, @Cpp.long_long.as.EqWith.impl.e41(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a699b9.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.f10080.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.a699b9.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.78fda2.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.b96 = impl_witness_access constants.%EqWith.impl_witness.abe, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.870d75.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.6e28cf.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.1 = specific_constant imports.%Core.NotEqual.861, @Cpp.long_long.as.EqWith.impl.e41(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.870d75.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.35a231.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.870d75.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.6e28cf.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.428 = impl_witness_access constants.%OrderedWith.impl_witness.184, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.ac49c5.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.1 = specific_constant imports.%Core.Greater.1b9, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.7a8931.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.c3dee8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.ac49c5.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.18d = impl_witness_access constants.%OrderedWith.impl_witness.184, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.963b6f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.cca121.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.1 = specific_constant imports.%Core.Less.4f0, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.963b6f.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.713f6d.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.963b6f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.cca121.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.e5e = impl_witness_access constants.%OrderedWith.impl_witness.184, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4ba090.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.1 = specific_constant imports.%Core.GreaterOrEquivalent.093, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.7749a7.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.241850.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4ba090.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.90c = impl_witness_access constants.%OrderedWith.impl_witness.184, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ccb3d.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.1 = specific_constant imports.%Core.LessOrEquivalent.602, @Cpp.long_long.as.OrderedWith.impl.f88(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.f5eed7.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ce310a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.cc9) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5ccb3d.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.e35: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.d70: %ImplicitAs.type.a57 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.8cb655.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.a5e(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.21260e.1: %T.as_type.as.EqWith.impl.NotEqual.type.8cb655.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.e22fe8.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.a5e(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f695c6.1: %T.as_type.as.EqWith.impl.Equal.type.e22fe8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.8cb655.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.a5e(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.21260e.2: %T.as_type.as.EqWith.impl.NotEqual.type.8cb655.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.e22fe8.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.a5e(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f695c6.2: %T.as_type.as.EqWith.impl.Equal.type.e22fe8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4b2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9d1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecb: %ImplicitAs.type.a57 = facet_value %i64, (%ImplicitAs.impl_witness.4b2) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.65d: <witness> = impl_witness imports.%EqWith.impl_witness_table.701, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.18dfc7.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6f6f7b.1: %T.as_type.as.EqWith.impl.Equal.type.18dfc7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.7c9a96.1: %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.18dfc7.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6f6f7b.2: %T.as_type.as.EqWith.impl.Equal.type.18dfc7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.7c9a96.2: %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.f5c: %EqWith.type.e35 = facet_value %i64, (%EqWith.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.7aa: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.f5c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.91d: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.f5c) [concrete]
// CHECK:STDOUT:   %.b54: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.7aa, %EqWith.facet.f5c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.4a5ff2.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6f6f7b.2, @T.as_type.as.EqWith.impl.Equal.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.4a5ff2.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6f6f7b.1, @T.as_type.as.EqWith.impl.Equal.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.96a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.96a, %ImplicitAs.facet.ecb [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.280: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.91d, %EqWith.facet.f5c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.2bdbd1.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.7c9a96.2, @T.as_type.as.EqWith.impl.NotEqual.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.2bdbd1.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.7c9a96.1, @T.as_type.as.EqWith.impl.NotEqual.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b28: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.297c9a.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.814536.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d5b800.1: %T.as_type.as.OrderedWith.impl.Greater.type.814536.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.44be43.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.ceb779.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c28535.1: %T.as_type.as.OrderedWith.impl.Less.type.ceb779.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.297c9a.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.814536.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d5b800.2: %T.as_type.as.OrderedWith.impl.Greater.type.814536.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.44be43.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.ceb779.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.17b(%T.d70) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c28535.2: %T.as_type.as.OrderedWith.impl.Less.type.ceb779.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f53: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.101, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.5613be.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.b76dd6.1: %T.as_type.as.OrderedWith.impl.Less.type.5613be.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a1a38b.1: %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.5613be.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.b76dd6.2: %T.as_type.as.OrderedWith.impl.Less.type.5613be.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a1a38b.2: %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.abe: %OrderedWith.type.b28 = facet_value %i64, (%OrderedWith.impl_witness.f53) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.3d7: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.abe) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.37d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.abe) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.43f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.abe) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.b92: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.abe) [concrete]
// CHECK:STDOUT:   %.15a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.43f, %OrderedWith.facet.abe [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.e877c3.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a1a38b.2, @T.as_type.as.OrderedWith.impl.Greater.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.e877c3.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a1a38b.1, @T.as_type.as.OrderedWith.impl.Greater.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.3d7, %OrderedWith.facet.abe [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d05b82.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.b76dd6.2, @T.as_type.as.OrderedWith.impl.Less.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d05b82.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.b76dd6.1, @T.as_type.as.OrderedWith.impl.Less.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.4ed: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.b92, %OrderedWith.facet.abe [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5e422c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5e422c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.327: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.37d, %OrderedWith.facet.abe [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.119ba6.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.119ba6.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.f16: <witness> = impl_witness imports.%EqWith.impl_witness_table.701, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.eda188.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.6, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.8b24a0.1: %T.as_type.as.EqWith.impl.Equal.type.eda188.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.6, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.dce29c.1: %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.eda188.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.5, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.8b24a0.2: %T.as_type.as.EqWith.impl.Equal.type.eda188.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.5, @T.as_type.as.EqWith.impl.a5e(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.dce29c.2: %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.146: %EqWith.type.e35 = facet_value Core.IntLiteral, (%EqWith.impl_witness.f16) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.9b3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.146) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.921: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.146) [concrete]
// CHECK:STDOUT:   %.01d: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.9b3, %EqWith.facet.146 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.f7e3db.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.8b24a0.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.8b24a0.2, @T.as_type.as.EqWith.impl.Equal.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.9a7805.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.f7e3db.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.8b24a0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.8b24a0.1, @T.as_type.as.EqWith.impl.Equal.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.9a7805.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.2 [concrete]
// CHECK:STDOUT:   %.740: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.921, %EqWith.facet.146 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.1ed807.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.dce29c.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.dce29c.2, @T.as_type.as.EqWith.impl.NotEqual.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.86e496.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.1ed807.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.dce29c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.dce29c.1, @T.as_type.as.EqWith.impl.NotEqual.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.86e496.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2e8: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.101, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.3a037c.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.3b87ee.1: %T.as_type.as.OrderedWith.impl.Less.type.3a037c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.1: %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.3a037c.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.3b87ee.2: %T.as_type.as.OrderedWith.impl.Less.type.3a037c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.2: %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5, @T.as_type.as.OrderedWith.impl.17b(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.207: %OrderedWith.type.b28 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.2e8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.14e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.207) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.414: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.207) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.574: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.207) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.b08: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.207) [concrete]
// CHECK:STDOUT:   %.68d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.574, %OrderedWith.facet.207 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.f81b1d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.2, @T.as_type.as.OrderedWith.impl.Greater.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.f9770f.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.f81b1d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.bb1ee5.1, @T.as_type.as.OrderedWith.impl.Greater.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.f9770f.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.2 [concrete]
// CHECK:STDOUT:   %.05d: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.14e, %OrderedWith.facet.207 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.cd419a.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.3b87ee.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.3b87ee.2, @T.as_type.as.OrderedWith.impl.Less.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.b86a28.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.cd419a.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.3b87ee.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.3b87ee.1, @T.as_type.as.OrderedWith.impl.Less.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.b86a28.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.2 [concrete]
// CHECK:STDOUT:   %.908: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.b08, %OrderedWith.facet.207 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f0a480.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.49e84d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f0a480.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.49e84d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.2 [concrete]
// CHECK:STDOUT:   %.06d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.414, %OrderedWith.facet.207 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.189c49.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.20035e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.189c49.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(%ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %bound_method.20035e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.0b53: @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.e22fe8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.f695c6.1)]
// CHECK:STDOUT:   %Core.import_ref.3b87: @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.8cb655.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.21260e.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.701 = impl_witness_table (%Core.import_ref.0b53, %Core.import_ref.3b87), @T.as_type.as.EqWith.impl.a5e [concrete]
// CHECK:STDOUT:   %Core.Equal.cdf: @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.e22fe8.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.f695c6.2)]
// CHECK:STDOUT:   %Core.NotEqual.c92: @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.8cb655.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.a5e.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.21260e.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9d1 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.777: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.ceb779.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.c28535.1)]
// CHECK:STDOUT:   %Core.import_ref.0cde: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.44be43.1)]
// CHECK:STDOUT:   %Core.import_ref.ac03: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.814536.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.d5b800.1)]
// CHECK:STDOUT:   %Core.import_ref.6ca1: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.297c9a.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.101 = impl_witness_table (%Core.import_ref.777, %Core.import_ref.0cde, %Core.import_ref.ac03, %Core.import_ref.6ca1), @T.as_type.as.OrderedWith.impl.17b [concrete]
// CHECK:STDOUT:   %Core.Less.50a: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.ceb779.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.c28535.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.c65: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f8e9c1.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.44be43.2)]
// CHECK:STDOUT:   %Core.Greater.018: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.814536.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.d5b800.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.054: @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.6e4707.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.17b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.297c9a.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.b54 = impl_witness_access constants.%EqWith.impl_witness.65d, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f7b.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.4a5ff2.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %T.as_type.as.EqWith.impl.Equal.type.18dfc7.1 = specific_constant imports.%Core.Equal.cdf, @T.as_type.as.EqWith.impl.a5e(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f7b.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.18dfc7.1 = name_ref Equal, %.loc10_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f7b.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.4a5ff2.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.280 = impl_witness_access constants.%EqWith.impl_witness.65d, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.7c9a96.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.2bdbd1.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.1 = specific_constant imports.%Core.NotEqual.c92, @T.as_type.as.EqWith.impl.a5e(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.7c9a96.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.84b1e2.1 = name_ref NotEqual, %.loc11_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.7c9a96.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.2bdbd1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.15a = impl_witness_access constants.%OrderedWith.impl_witness.f53, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a1a38b.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.e877c3.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.1 = specific_constant imports.%Core.Greater.018, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a1a38b.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.69ac51.1 = name_ref Greater, %.loc12_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a1a38b.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.e877c3.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.89c = impl_witness_access constants.%OrderedWith.impl_witness.f53, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b76dd6.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d05b82.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %T.as_type.as.OrderedWith.impl.Less.type.5613be.1 = specific_constant imports.%Core.Less.50a, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b76dd6.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.5613be.1 = name_ref Less, %.loc13_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b76dd6.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d05b82.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.4ed = impl_witness_access constants.%OrderedWith.impl_witness.f53, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5e422c.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.1 = specific_constant imports.%Core.GreaterOrEquivalent.054, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.72d52a.1 = name_ref GreaterOrEquivalent, %.loc14_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f859af.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5e422c.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.327 = impl_witness_access constants.%OrderedWith.impl_witness.f53, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.119ba6.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.1 = specific_constant imports.%Core.LessOrEquivalent.c65, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.fff669.1 = name_ref LessOrEquivalent, %.loc15_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6d2dd5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(constants.%ImplicitAs.facet.ecb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.119ba6.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.01d = impl_witness_access constants.%EqWith.impl_witness.f16, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.8b24a0.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.f7e3db.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.9a7805.1]
// CHECK:STDOUT:   %.loc17_5.3: %T.as_type.as.EqWith.impl.Equal.type.eda188.1 = specific_constant imports.%Core.Equal.cdf, @T.as_type.as.EqWith.impl.a5e(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.8b24a0.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.eda188.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.8b24a0.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.f7e3db.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.faa4cf.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.9a7805.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.740 = impl_witness_access constants.%EqWith.impl_witness.f16, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.dce29c.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.1ed807.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.86e496.1]
// CHECK:STDOUT:   %.loc18_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.1 = specific_constant imports.%Core.NotEqual.c92, @T.as_type.as.EqWith.impl.a5e(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.dce29c.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.c69aed.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.dce29c.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.1ed807.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.1064c0.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.86e496.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.68d = impl_witness_access constants.%OrderedWith.impl_witness.2e8, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bb1ee5.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.f81b1d.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.f9770f.1]
// CHECK:STDOUT:   %.loc19_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.1 = specific_constant imports.%Core.Greater.018, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bb1ee5.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.847fb5.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bb1ee5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.f81b1d.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.1552dd.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.f9770f.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.05d = impl_witness_access constants.%OrderedWith.impl_witness.2e8, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3b87ee.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.cd419a.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.b86a28.1]
// CHECK:STDOUT:   %.loc20_5.3: %T.as_type.as.OrderedWith.impl.Less.type.3a037c.1 = specific_constant imports.%Core.Less.50a, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3b87ee.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.3a037c.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3b87ee.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.cd419a.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.d03a09.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.b86a28.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.908 = impl_witness_access constants.%OrderedWith.impl_witness.2e8, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f0a480.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.49e84d.1]
// CHECK:STDOUT:   %.loc21_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.1 = specific_constant imports.%Core.GreaterOrEquivalent.054, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.34cd2b.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.216ae0.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f0a480.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.785ab2.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.49e84d.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.06d = impl_witness_access constants.%OrderedWith.impl_witness.2e8, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.189c49.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.cc9]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.5(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.20035e.1]
// CHECK:STDOUT:   %.loc22_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.1 = specific_constant imports.%Core.LessOrEquivalent.c65, @T.as_type.as.OrderedWith.impl.17b(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.c47891.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.480429.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.189c49.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.6(constants.%ImplicitAs.facet.cc9) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.82f3e7.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.20035e.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.98e: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.15d: %T.as_type.as.ImplicitAs.impl.Convert.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.8a9: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.d70: %ImplicitAs.type.a57 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1ae: <witness> = impl_witness imports.%Copy.impl_witness_table.d10 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.1ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.26f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f48, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.030: %ImplicitAs.type.a57 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.26f) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.0d5: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.f2f, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.85e: %AddAssignWith.type.8a9 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.0d5) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.95c: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet.85e) [concrete]
// CHECK:STDOUT:   %.9d2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.95c, %AddAssignWith.facet.85e [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.48fc4a.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.48fc4a.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.143: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.726c91.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.726c91.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.5a8: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.1c3, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.9f8: %SubAssignWith.type.143 = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.5a8) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.762: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet.9f8) [concrete]
// CHECK:STDOUT:   %.0fb: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.762, %SubAssignWith.facet.9f8 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.f50fea.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.f50fea.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.6ad: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.955: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.fa8, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.51f: %MulAssignWith.type.6ad = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.955) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.58e: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet.51f) [concrete]
// CHECK:STDOUT:   %.fdf: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.58e, %MulAssignWith.facet.51f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.6524e5.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.6524e5.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.309: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.604: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.5fc, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.0cb: %DivAssignWith.type.309 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.604) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.9db: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet.0cb) [concrete]
// CHECK:STDOUT:   %.f87: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.9db, %DivAssignWith.facet.0cb [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.c86a2f.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.c86a2f.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.a99: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.0e8: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.23d, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.a35: %ModAssignWith.type.a99 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.0e8) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.712: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet.a35) [concrete]
// CHECK:STDOUT:   %.da57: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.712, %ModAssignWith.facet.a35 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2a21fb.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2a21fb.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.5fd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.49b: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.f6e, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.2ed: %BitAndAssignWith.type.5fd = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.49b) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.3ce: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet.2ed) [concrete]
// CHECK:STDOUT:   %.6ab: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.3ce, %BitAndAssignWith.facet.2ed [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.7586ea.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.7586ea.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.f53: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.a74: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.c57, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.dd9: %BitOrAssignWith.type.f53 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.a74) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.b0b: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet.dd9) [concrete]
// CHECK:STDOUT:   %.c35: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.b0b, %BitOrAssignWith.facet.dd9 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.2e4590.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.2e4590.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.1cb: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.e0e: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.94b, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.acf: %BitXorAssignWith.type.1cb = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.e0e) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.3d4: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet.acf) [concrete]
// CHECK:STDOUT:   %.06c: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.3d4, %BitXorAssignWith.facet.acf [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.03b326.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.03b326.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.8aa: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.377: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.c11, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.9ab: %LeftShiftAssignWith.type.8aa = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.377) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.ba7: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet.9ab) [concrete]
// CHECK:STDOUT:   %.da55: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.ba7, %LeftShiftAssignWith.facet.9ab [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.28b62f.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.28b62f.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.c92: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.9bf: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.fdf, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.112: %RightShiftAssignWith.type.c92 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.9bf) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.937: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet.112) [concrete]
// CHECK:STDOUT:   %.a5c: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.937, %RightShiftAssignWith.facet.112 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ab7395.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ab7395.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.030) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a2d: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.98e) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.15d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f48 = impl_witness_table (%Core.import_ref.a2d), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.718: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.f2f = impl_witness_table (%Core.import_ref.718), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a64: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d10 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.794: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.726c91.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.1c3 = impl_witness_table (%Core.import_ref.794), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.058: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.726c91.2)]
// CHECK:STDOUT:   %Core.import_ref.a09: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.fa8 = impl_witness_table (%Core.import_ref.a09), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e3a: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.2)]
// CHECK:STDOUT:   %Core.import_ref.590: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.5fc = impl_witness_table (%Core.import_ref.590), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.406: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.2)]
// CHECK:STDOUT:   %Core.import_ref.6fe: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.23d = impl_witness_table (%Core.import_ref.6fe), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fcd: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.2)]
// CHECK:STDOUT:   %Core.import_ref.5b6: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.f6e = impl_witness_table (%Core.import_ref.5b6), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.65c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.2)]
// CHECK:STDOUT:   %Core.import_ref.428: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.c57 = impl_witness_table (%Core.import_ref.428), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d64: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.2)]
// CHECK:STDOUT:   %Core.import_ref.fcc: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.94b = impl_witness_table (%Core.import_ref.fcc), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.32a: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.2)]
// CHECK:STDOUT:   %Core.import_ref.a22: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.c11 = impl_witness_table (%Core.import_ref.a22), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c97: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.2)]
// CHECK:STDOUT:   %Core.import_ref.477: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.fdf = impl_witness_table (%Core.import_ref.477), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.876: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.9d2 = impl_witness_access constants.%AddAssignWith.impl_witness.0d5, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.48fc4a.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.1 = specific_constant imports.%Core.Op.a64, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.b4dd51.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0e05f5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.48fc4a.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.0fb = impl_witness_access constants.%SubAssignWith.impl_witness.5a8, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.f50fea.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.1 = specific_constant imports.%Core.Op.058, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.59c3e0.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.fa0f8d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.f50fea.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.fdf = impl_witness_access constants.%MulAssignWith.impl_witness.955, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.6524e5.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.1 = specific_constant imports.%Core.Op.e3a, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.e02657.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.b6e63f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.6524e5.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.f87 = impl_witness_access constants.%DivAssignWith.impl_witness.604, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.c86a2f.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.1 = specific_constant imports.%Core.Op.406, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.2f9287.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bfaef9.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.c86a2f.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.da57 = impl_witness_access constants.%ModAssignWith.impl_witness.0e8, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2a21fb.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.1 = specific_constant imports.%Core.Op.fcd, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.50e75c.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.aabe3c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.2a21fb.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.6ab = impl_witness_access constants.%BitAndAssignWith.impl_witness.49b, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.7586ea.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.1 = specific_constant imports.%Core.Op.65c, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.5b0e84.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.d3e329.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.7586ea.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.c35 = impl_witness_access constants.%BitOrAssignWith.impl_witness.a74, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.2e4590.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.1 = specific_constant imports.%Core.Op.d64, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.f56bd9.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.4890e6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.2e4590.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.06c = impl_witness_access constants.%BitXorAssignWith.impl_witness.e0e, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.03b326.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.1 = specific_constant imports.%Core.Op.32a, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.ee589f.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.46b8cc.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.03b326.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.da55 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.377, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.28b62f.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.1 = specific_constant imports.%Core.Op.c97, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.e3eece.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.76b6bf.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.28b62f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.a5c = impl_witness_access constants.%RightShiftAssignWith.impl_witness.9bf, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.a57 = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.26f) [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.a57 = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.030]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ab7395.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.1 = specific_constant imports.%Core.Op.876, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.cab567.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.c6ebc3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.030) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.ab7395.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.dd2: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.d70: %ImplicitAs.type.a57 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4b2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9d1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecb: %ImplicitAs.type.a57 = facet_value %i64, (%ImplicitAs.impl_witness.4b2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.37d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.f2f, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.a0f: %AddAssignWith.type.dd2 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.37d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.095: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.a0f) [concrete]
// CHECK:STDOUT:   %.02b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.095, %AddAssignWith.facet.a0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.96a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.96a, %ImplicitAs.facet.ecb [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.86e: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.726c91.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.726c91.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.6c3: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.1c3, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.f56: %SubAssignWith.type.86e = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.6c3) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.044: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet.f56) [concrete]
// CHECK:STDOUT:   %.303: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.044, %SubAssignWith.facet.f56 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.1286dd.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.1286dd.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.f20: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.c47: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.fa8, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.f43: %MulAssignWith.type.f20 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.c47) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.499: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet.f43) [concrete]
// CHECK:STDOUT:   %.8c6: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.499, %MulAssignWith.facet.f43 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.150281.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.150281.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.afe: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.537: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.5fc, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.034: %DivAssignWith.type.afe = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.537) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.099: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet.034) [concrete]
// CHECK:STDOUT:   %.204: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.099, %DivAssignWith.facet.034 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6b3a45.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6b3a45.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2bf: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.915: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.23d, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.9c9: %ModAssignWith.type.2bf = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.915) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.d3c: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet.9c9) [concrete]
// CHECK:STDOUT:   %.9fc: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.d3c, %ModAssignWith.facet.9c9 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.d86284.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.d86284.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.bad: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.600: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.f6e, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.45f: %BitAndAssignWith.type.bad = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.600) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.0fd: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet.45f) [concrete]
// CHECK:STDOUT:   %.1ac: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.0fd, %BitAndAssignWith.facet.45f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99eac2.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99eac2.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.a61: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.2ee: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.c57, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.9a0: %BitOrAssignWith.type.a61 = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.2ee) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.ae3: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet.9a0) [concrete]
// CHECK:STDOUT:   %.3ee: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.ae3, %BitOrAssignWith.facet.9a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.25abd8.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.25abd8.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.8af: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.691: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.94b, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.63f: %BitXorAssignWith.type.8af = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.691) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.92e: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet.63f) [concrete]
// CHECK:STDOUT:   %.c51: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.92e, %BitXorAssignWith.facet.63f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.ca7c4e.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.ca7c4e.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.1e6: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.f82: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.c11, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.957: %LeftShiftAssignWith.type.1e6 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.f82) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.d8c: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet.957) [concrete]
// CHECK:STDOUT:   %.ebd: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.d8c, %LeftShiftAssignWith.facet.957 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f93991.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f93991.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f11: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.2fa: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.fdf, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.ca3: %RightShiftAssignWith.type.f11 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.ce3: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet.ca3) [concrete]
// CHECK:STDOUT:   %.b05: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.ce3, %RightShiftAssignWith.facet.ca3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1b45c4.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1b45c4.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.718: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.f2f = impl_witness_table (%Core.import_ref.718), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a64: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9d1 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.794: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.726c91.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.1c3 = impl_witness_table (%Core.import_ref.794), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.058: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.9980fa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.726c91.2)]
// CHECK:STDOUT:   %Core.import_ref.a09: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.fa8 = impl_witness_table (%Core.import_ref.a09), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e3a: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.84ad55.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.d6b6aa.2)]
// CHECK:STDOUT:   %Core.import_ref.590: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.5fc = impl_witness_table (%Core.import_ref.590), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.406: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.d93afb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.c0ddb1.2)]
// CHECK:STDOUT:   %Core.import_ref.6fe: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.23d = impl_witness_table (%Core.import_ref.6fe), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fcd: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.d22cf1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.3d1ba5.2)]
// CHECK:STDOUT:   %Core.import_ref.5b6: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.f6e = impl_witness_table (%Core.import_ref.5b6), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.65c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.4ac1e9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.a31b6d.2)]
// CHECK:STDOUT:   %Core.import_ref.428: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.c57 = impl_witness_table (%Core.import_ref.428), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d64: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.944d5b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.3b7278.2)]
// CHECK:STDOUT:   %Core.import_ref.fcc: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.94b = impl_witness_table (%Core.import_ref.fcc), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.32a: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c2a862.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.da17ff.2)]
// CHECK:STDOUT:   %Core.import_ref.a22: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.c11 = impl_witness_table (%Core.import_ref.a22), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c97: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.674771.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.74aebe.2)]
// CHECK:STDOUT:   %Core.import_ref.477: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.fdf = impl_witness_table (%Core.import_ref.477), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.876: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.f5b3db.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.0019d5.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.02b = impl_witness_access constants.%AddAssignWith.impl_witness.37d, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = specific_constant imports.%Core.Op.a64, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.303 = impl_witness_access constants.%SubAssignWith.impl_witness.6c3, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.1286dd.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.1 = specific_constant imports.%Core.Op.058, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.268ac6.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1c90c3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.1286dd.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.8c6 = impl_witness_access constants.%MulAssignWith.impl_witness.c47, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.150281.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.1 = specific_constant imports.%Core.Op.e3a, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.04b78a.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.01eb45.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.150281.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.204 = impl_witness_access constants.%DivAssignWith.impl_witness.537, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6b3a45.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.1 = specific_constant imports.%Core.Op.406, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.071de1.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.bf60f0.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.6b3a45.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.9fc = impl_witness_access constants.%ModAssignWith.impl_witness.915, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.d86284.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.1 = specific_constant imports.%Core.Op.fcd, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.d9c1a8.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.7f5a33.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.d86284.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.1ac = impl_witness_access constants.%BitAndAssignWith.impl_witness.600, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc16_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc16_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc16_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99eac2.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.1 = specific_constant imports.%Core.Op.65c, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.36cbed.1 = name_ref Op, %.loc16_5.3 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.69ba22.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.99eac2.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.3ee = impl_witness_access constants.%BitOrAssignWith.impl_witness.2ee, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.25abd8.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.1 = specific_constant imports.%Core.Op.d64, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.cdfc75.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.54913d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.25abd8.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.c51 = impl_witness_access constants.%BitXorAssignWith.impl_witness.691, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.ca7c4e.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.1 = specific_constant imports.%Core.Op.32a, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.d556a4.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.ad5f91.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.ca7c4e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.ebd = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.f82, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f93991.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.1 = specific_constant imports.%Core.Op.c97, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.eff9ce.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.ba908d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.f93991.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.b05 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.2fa, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1b45c4.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.1 = specific_constant imports.%Core.Op.876, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.b878fc.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.8c132c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.1b45c4.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long_long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.144: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.d70: %ImplicitAs.type.a57 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.ddb: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.f2f, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.c4c: %AddAssignWith.type.144 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.ddb) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.ca9: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.c4c) [concrete]
// CHECK:STDOUT:   %.d0b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.ca9, %AddAssignWith.facet.c4c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fd006b.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fd006b.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.718: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.f2f = impl_witness_table (%Core.import_ref.718), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a64: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.d0b = impl_witness_access constants.%AddAssignWith.impl_witness.ddb, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.f09) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.a57 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fd006b.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.1 = specific_constant imports.%Core.Op.a64, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.bbde48.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.cf7a8d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.fd006b.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.3(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_runtime_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cc9: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.cc9) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet.cc9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6d9 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.122: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5f8: %ImplicitAs.type.270 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.122) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.de7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.5f8) [concrete]
// CHECK:STDOUT:   %.19a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.de7, %ImplicitAs.facet.5f8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.4f6: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.dd2: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.d70: %ImplicitAs.type.a57 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.d70) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4b2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9d1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecb: %ImplicitAs.type.a57 = facet_value %i64, (%ImplicitAs.impl_witness.4b2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.37d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.f2f, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.a0f: %AddAssignWith.type.dd2 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.37d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.095: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.a0f) [concrete]
// CHECK:STDOUT:   %.02b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.095, %AddAssignWith.facet.a0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.96a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.ecb) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.96a, %ImplicitAs.facet.ecb [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.718: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.f2f = impl_witness_table (%Core.import_ref.718), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a64: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.51e002.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.6dc61b.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9d1 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.19a = impl_witness_access constants.%ImplicitAs.impl_witness.122, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.64f]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %b: %i64 = wrapper_binding b, %.loc9_16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6d9 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.02b = impl_witness_access constants.%AddAssignWith.impl_witness.37d, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.a57 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.4b2) [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.a57 = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.ecb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = specific_constant imports.%Core.Op.a64, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.f90187.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.dddd1a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.ecb) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.767959.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.8fd = impl_witness_access constants.%ImplicitAs.impl_witness.4b2, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a57: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6f8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a57 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f09) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.01f: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.8df: type = fn_type_with_self_type %Inc.WithSelf.Op.type.01f, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7e1: %pattern_type.d58 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.718: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt.7e1 [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.3d8: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.7f0: type = fn_type_with_self_type %Dec.WithSelf.Op.type.3d8, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.type: type = fn_type @Cpp.long_long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op: %Cpp.long_long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc8_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.dc9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long_long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d58 = ref_param_pattern [concrete = constants.%self.param_patt.7e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d58 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2be = impl_witness_access constants.%ImplicitAs.impl_witness.f09, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.8df = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.7f0 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.1(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.1(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.2(%self.param: ref %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.2(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long.patt: %pattern_type.4cd = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b88: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.615: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ac [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1f0: %ImplicitAs.type.b88 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a08: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.1f0) [concrete]
// CHECK:STDOUT:   %.c23: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a08, %ImplicitAs.facet.1f0 [concrete]
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
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %carbon_u64.patt: %pattern_type.878 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.edd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bf2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0ef: %ImplicitAs.type.615 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a86) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b73: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.0ef) [concrete]
// CHECK:STDOUT:   %.072: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b73, %ImplicitAs.facet.0ef [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4bb: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d00: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97f: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %carbon_u64_result.patt: %pattern_type.97f = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   %ptr.e40: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.774: type = facet_type <@As, @As(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.104: <witness> = impl_witness imports.%As.impl_witness_table.16f [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.774 = facet_value Core.IntLiteral, (%As.impl_witness.104) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e17: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.44c: type = fn_type_with_self_type %As.WithSelf.Convert.type.e17, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.256: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5db [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fbf: %ImplicitAs.type.5be = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.256) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.fbf) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da0, %ImplicitAs.facet.fbf [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.877, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.984: type = class_type @Array, @Array(%f32.2d9, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.32f: type = pattern_type %Array.984 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0f1, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec7: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet.ec7) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet.ec7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.29b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.984 = tuple_value (%float.3a8) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.ab7: %pattern_type.afb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20c: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt.ab7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3dc: %pattern_type.32f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e84: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt.3dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.116: %pattern_type.97f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.eb6: %pattern_type.97f = wrapper_binding_pattern self, %self.param_patt.116 [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2ef: %pattern_type.97e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a5a: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt.2ef [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Array = %Core.Array
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ac = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.b36 [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.602: @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.edd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bf2 = impl_witness_table (%Core.import_ref.602), @Core.IntLiteral.as.ImplicitAs.impl.7db [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.16f = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.163 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5db = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long_long.as.ImplicitAs.impl.540 [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/types/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.decl: %ULongResult.cpp_destructor.type = fn_decl @ULongResult.cpp_destructor [concrete = constants.%ULongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.decl: %ULongLongResult.cpp_destructor.type = fn_decl @ULongLongResult.cpp_destructor [concrete = constants.%ULongLongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc24_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.32f = ref_param_pattern [concrete = constants.%self.param_patt.3dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.32f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e84]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.984 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.984 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97f = ref_param_pattern [concrete = constants.%self.param_patt.116]
// CHECK:STDOUT:     %self.patt: %pattern_type.97f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eb6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ULongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ULongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc19 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97e = ref_param_pattern [concrete = constants.%self.param_patt.2ef]
// CHECK:STDOUT:     %self.patt: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ULongLongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ULongLongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.c23 = impl_witness_access constants.%ImplicitAs.impl_witness.c73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc21: %.072 = impl_witness_access constants.%ImplicitAs.impl_witness.a86, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e2]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.4bb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_u64: %u64 = wrapper_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.878 = value_binding_pattern carbon_u64 [concrete = constants.%carbon_u64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc24_64.1: <bound method> = bound_method %float, %impl.elem0.loc24_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_64.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.2d9 = call %bound_method.loc24_64.2(%float) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_64.2: init %f32.2d9 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_64.3: ref %f32.2d9 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_64.4: init %f32.2d9 to %.loc24_64.3 = in_place_init %.loc24_64.2 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_64.5: init %Array.984 to %a.var = array_init (%.loc24_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %Array.984 = converted %.loc24_64.1, %.loc24_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_55: type = splice_block %Array [concrete = constants.%Array.984] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc24: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.44c = impl_witness_access constants.%As.impl_witness.104, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.256, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.2d9, constants.%int_1.5b8) [concrete = constants.%Array.984]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.984 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.32f = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.32f = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%a.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_79.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_79.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_105.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_105.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_107.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16(%.loc16_107.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.1(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.1(%self.param: ref %f32.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24: ref %f32.2d9 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.2(%self.param: ref %Array.984) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.2(%self.param: ref %Array.984) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22(%self.param: ref %ULongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22(%self.param: ref %ULongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%ULongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc19(%self.param: ref %ULongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc19(%self.param: ref %ULongLongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%ULongLongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.4cd = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.4cd = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b88: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ac [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b88 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a08: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c23: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a08, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.877: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.4cd = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.4cd = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2e6: <witness> = impl_witness imports.%Copy.impl_witness_table.6d7 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.2e6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.074: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %Copy.WithSelf.Op.type.074, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %self.param_patt.4cb: %pattern_type.878 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.198: %pattern_type.878 = wrapper_binding_pattern self, %self.param_patt.4cb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.1ea: %pattern_type.4cd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.919: %pattern_type.4cd = wrapper_binding_pattern self, %self.param_patt.1ea [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ac = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.b36 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.6d7 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.878 = ref_param_pattern [concrete = constants.%self.param_patt.4cb]
// CHECK:STDOUT:     %self.patt: %pattern_type.878 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.198]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.878 = ref_param_pattern [concrete = constants.%self.param_patt.4cb]
// CHECK:STDOUT:     %self.patt: %pattern_type.878 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.198]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4cd = ref_param_pattern [concrete = constants.%self.param_patt.1ea]
// CHECK:STDOUT:     %self.patt: %pattern_type.4cd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.919]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.unsigned_long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.unsigned_long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4cd = ref_param_pattern [concrete = constants.%self.param_patt.1ea]
// CHECK:STDOUT:     %self.patt: %pattern_type.4cd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.919]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.unsigned_long_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.unsigned_long_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.c23 = impl_witness_access constants.%ImplicitAs.impl_witness.c73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.637 = impl_witness_access constants.%Copy.impl_witness.2e6, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.unsigned_long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.unsigned_long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.unsigned_long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
