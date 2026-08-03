// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=arm64-apple-darwin
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.darwin.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.darwin.carbon

// 64-bit Darwin is a bit unusual: it's an LP64 target but its `int64_t` is
// `long long`. So for type mapping purposes:
//
// - `int` <-> `i32` == `Core.Int(32)` (of course).
// - `long` <-> `Cpp.long` == `Core.CppCompat.Long64`.
// - `long long` <-> `Cpp.long_long` == `Core.Int(64)`.

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

  let cpp_long: Cpp.long = 1;
  let unused cpp_long_result: Cpp.LongResult = Cpp.PassLongLong(cpp_long);

  let carbon_i64: i64 = cpp_long_long;
  let unused carbon_i64_result: Cpp.LongLongResult = Cpp.PassLongLong(carbon_i64);

  var unused a: array(f32, 1 as Cpp.long_long) = (1.0,);
  //@dump-sem-ir-end
}


// --- fail_implicit_conversion_i64_to_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let carbon_i64: i64 = 1;
  // CHECK:STDERR: fail_implicit_conversion_i64_to_long.carbon:[[@LINE+7]]:35: error: cannot implicitly convert expression of type `i64` to `Cpp.long` [ConversionFailure]
  // CHECK:STDERR:   let unused cpp_long: Cpp.long = carbon_i64;
  // CHECK:STDERR:                                   ^~~~~~~~~~
  // CHECK:STDERR: fail_implicit_conversion_i64_to_long.carbon:[[@LINE+4]]:35: note: type `i64` does not implement interface `Core.ImplicitAs(Cpp.long)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused cpp_long: Cpp.long = carbon_i64;
  // CHECK:STDERR:                                   ^~~~~~~~~~
  // CHECK:STDERR:
  let unused cpp_long: Cpp.long = carbon_i64;
}

// --- copy_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyLongLong() {
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
  let b: i64 = 1;
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
  let b: i64 = 1;
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

// --- fail_todo_comparisons_heterogeneous_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongAndI32() {
  let a: Cpp.long = 1;
  let b: i32 = 1;
  a == b;
  b == a;
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

// --- arithmetic_homogeneous_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn ArithmeticHomogeneousLong() {
  let x: Cpp.long = 1;
  let y: Cpp.long = 1;

  AssertSameType(-x, x);
  AssertSameType(x + y, x);
  AssertSameType(x - y, x);
  AssertSameType(x * y, x);
  AssertSameType(x / y, x);
  AssertSameType(x % y, x);
}

// --- arithmetic_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLeftSide() {
  let b: i64 = 1;
  let x: Cpp.long = 1;

  AssertSameType(x + b, b);
  AssertSameType(x - b, b);
  AssertSameType(x * b, b);
  AssertSameType(x / b, b);
  AssertSameType(x % b, b);

  AssertSameType(x + 1, x);
  AssertSameType(x - 1, x);
  AssertSameType(x * 1, x);
  AssertSameType(x / 1, x);
  AssertSameType(x % 1, x);
}

// --- arithmetic_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongRightSide() {
  let b: i64 = 1;
  let x: Cpp.long = 1;
  AssertSameType(b + x, b);
  AssertSameType(b - x, b);
  AssertSameType(b * x, b);
  AssertSameType(b / x, b);
  AssertSameType(b % x, b);

  AssertSameType(1 + x, x);
  AssertSameType(1 - x, x);
  AssertSameType(1 * x, x);
  AssertSameType(1 / x, x);
  AssertSameType(1 % x, x);
}

// --- fail_todo_arithmetic_heterogeneous_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongAndI32() {
  let x: Cpp.long = 1;
  let y: i32 = 1;
  AssertSameType(x + y, x);

  AssertSameType(y + x, x);
}

// --- fail_todo_arithmetic_heterogeneous_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn BitWiseHomogeneousLong() {
  let a: Cpp.long = 1;
  let b: Cpp.long = 1;

  AssertSameType(^a, a);
  AssertSameType(a & b, a);
  AssertSameType(a | b, a);
  AssertSameType(a ^ b, a);
  AssertSameType(a << b, a);
  AssertSameType(a >> b, a);
}

// --- bitwise_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLeftSide() {
  let b: i64 = 1;
  let a: Cpp.long = 1;

  AssertSameType(a & b, b);
  AssertSameType(a | b, b);
  AssertSameType(a ^ b, b);
  AssertSameType(a << b, b);
  AssertSameType(a >> b, b);

  AssertSameType(a & 1, a);
  AssertSameType(a | 1, a);
  AssertSameType(a ^ 1, a);
  AssertSameType(a << 1, a);
  AssertSameType(a >> 1, a);
}

// --- bitwise_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongRightSide() {
  let b: i64 = 1;
  let a: Cpp.long = 1;

  AssertSameType(b & a, b);
  AssertSameType(b | a, b);
  AssertSameType(b ^ a, b);
  AssertSameType(b << a, b);
  AssertSameType(b >> a, b);

  AssertSameType(1 & a, a);
  AssertSameType(1 | a, a);
  AssertSameType(1 ^ a, a);
  AssertSameType(1 << a, a);
  AssertSameType(1 >> a, a);
}

// --- bitwise_heterogeneous_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongAndI32() {
  let a: Cpp.long = 1;
  let b: i32 = 1;

  AssertSameType(a & b, a);
  AssertSameType(b & a, a);
}

// --- fail_todo_bitwise_heterogeneous_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T: type](unused a: T, unused b: T) {}

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

// --- compound_assignment_hereogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongAndI64() {
  var a: i64 = 1;
  //@dump-sem-ir-begin
  let b: Cpp.long = 1;
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

// --- compound_assignment_heterogeneous_long_and_runtime_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongAndRuntimeI64() {
  //@dump-sem-ir-begin
  var a: i64 = 1;
  let b: Cpp.long = 1;
  a += b;
  //@dump-sem-ir-end
}

// --- fail_todo_compound_assignment_heterogeneous_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousLongAndI32() {
  var a: Cpp.long = 1;
  a += (1 as i32);
}

// --- fail_compound_assignment_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousLongAndI64() {
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

  let cpp_unsigned_long: Cpp.unsigned_long = 1;
  let unused cpp_unsigned_long_result: Cpp.ULongResult = Cpp.PassULongLong(cpp_unsigned_long);

  let carbon_u64: u64 = 1;
  let unused carbon_u64_result: Cpp.ULongLongResult = Cpp.PassULongLong(carbon_u64);

  var unused a: array(f32, 1 as Cpp.unsigned_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- fail_pass_u64_to_ulong.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let carbon_u64: u64 = 1;

  // For portability reasons, this is not allowed even though `u64`
  // happens to fit in `unsigned long` on this target.
  // CHECK:STDERR: fail_pass_u64_to_ulong.carbon:[[@LINE+7]]:53: error: cannot implicitly convert expression of type `u64` to `Cpp.unsigned_long` [ConversionFailure]
  // CHECK:STDERR:   let unused cpp_unsigned_long: Cpp.unsigned_long = carbon_u64;
  // CHECK:STDERR:                                                     ^~~~~~~~~~
  // CHECK:STDERR: fail_pass_u64_to_ulong.carbon:[[@LINE+4]]:53: note: type `u64` does not implement interface `Core.ImplicitAs(Cpp.unsigned_long)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused cpp_unsigned_long: Cpp.unsigned_long = carbon_u64;
  // CHECK:STDERR:                                                     ^~~~~~~~~~
  // CHECK:STDERR:
  let unused cpp_unsigned_long: Cpp.unsigned_long = carbon_u64;
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


// CHECK:STDOUT: --- long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %cpp_long.patt: %pattern_type.ada = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %carbon_long.patt: %pattern_type.f57 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a3a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c71: %ImplicitAs.type.221 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.294: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.294, %ImplicitAs.facet.c71 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.c37: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.e4e: <witness> = impl_witness imports.%As.impl_witness_table.e98 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.c37 = facet_value Core.IntLiteral, (%As.impl_witness.e4e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.518: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.f4b: type = fn_type_with_self_type %As.WithSelf.Convert.type.518, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.79f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.851 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d18: %ImplicitAs.type.7cb = facet_value %Cpp.long, (%ImplicitAs.impl_witness.79f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.29c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.d18) [concrete]
// CHECK:STDOUT:   %.1e3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.29c, %ImplicitAs.facet.d18 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.33f, %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a3a = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.c8d [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.e98 = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.ed4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.851 = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.long.as.ImplicitAs.impl.b7c [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_28.2: %Cpp.long = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long: %Cpp.long = wrapper_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.ada = value_binding_pattern cpp_long [concrete = constants.%cpp_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %impl.elem0.loc9: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9(%cpp_long.ref)
// CHECK:STDOUT:   %.loc9_33.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_33.2: %i64 = converted %cpp_long.ref, %.loc9_33.1
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = wrapper_binding carbon_long, %.loc9_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.f57 = value_binding_pattern carbon_long [concrete = constants.%carbon_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
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
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.f4b = impl_witness_access constants.%As.impl_witness.e4e, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.1e3 = impl_witness_access constants.%ImplicitAs.impl_witness.79f, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7da: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.747: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.54e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dbc: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.747) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.57a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.dbc) [concrete]
// CHECK:STDOUT:   %.869: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.57a, %ImplicitAs.facet.dbc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.885: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2eb [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.728: %ImplicitAs.type.14f = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.885) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.798: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.728) [concrete]
// CHECK:STDOUT:   %.0ae: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.798, %ImplicitAs.facet.728 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.68f: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.a83: <witness> = impl_witness imports.%As.impl_witness_table.fcb [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.68f = facet_value Core.IntLiteral, (%As.impl_witness.a83) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d43: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %As.WithSelf.Convert.type.d43, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f9e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d48 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.020: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.f9e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a35: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.020) [concrete]
// CHECK:STDOUT:   %.1d2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a35, %ImplicitAs.facet.020 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.54e = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2eb = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.unsigned_long.as.ImplicitAs.impl.c4b [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.fcb = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.60e [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d48 = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.unsigned_long.as.ImplicitAs.impl.8de [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.869 = impl_witness_access constants.%ImplicitAs.impl_witness.747, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc8_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc8_46.2: %Cpp.unsigned_long = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = wrapper_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete = constants.%cpp_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %impl.elem0.loc9: %.0ae = impl_witness_access constants.%ImplicitAs.impl_witness.885, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_unsigned_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9: init %u64 = call %bound_method.loc9(%cpp_unsigned_long.ref)
// CHECK:STDOUT:   %.loc9_42.1: %u64 = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_42.2: %u64 = converted %cpp_unsigned_long.ref, %.loc9_42.1
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = wrapper_binding carbon_unsigned_long, %.loc9_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern carbon_unsigned_long [concrete = constants.%carbon_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
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
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.0ea = impl_witness_access constants.%As.impl_witness.a83, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.unsigned_long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.1d2 = impl_witness_access constants.%ImplicitAs.impl_witness.f9e, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %cpp_long_long.patt: %pattern_type.f57 = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.091: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6e8: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.091) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.6e8) [concrete]
// CHECK:STDOUT:   %.f34: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a04, %ImplicitAs.facet.6e8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c5e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d98: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.357: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %cpp_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.662: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %cpp_long.patt: %pattern_type.ada = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %cpp_long_result.patt: %pattern_type.ae7 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   %ptr.aec: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %carbon_i64.patt: %pattern_type.f57 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   %carbon_i64_result.patt: %pattern_type.357 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.998: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.f25(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.2e1: <witness> = impl_witness imports.%As.impl_witness_table.1c5, @Core.IntLiteral.as.As.impl.f25(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.1a1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.f25(%int_64) [concrete]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.2
// CHECK:STDOUT:     .LongLongResult = %LongLongResult.decl
// CHECK:STDOUT:     .PassLongLong = %PassLongLong.cpp_overload_set.value
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b8b = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.f25.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.f25.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.1c5 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl.f25 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.f34 = impl_witness_access constants.%ImplicitAs.impl_witness.091, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc15_38.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_38.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.c5e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_38.2(%int_1.loc15) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc15_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc15_38.2: %i64 = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = wrapper_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.f57 = value_binding_pattern cpp_long_long [concrete = constants.%cpp_long_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %i64 = name_ref cpp_long_long, %cpp_long_long
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
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc18: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_28.2: %Cpp.long = converted %int_1.loc18, %.loc18_28.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_20: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long: %Cpp.long = wrapper_binding cpp_long, %.loc18_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.ada = value_binding_pattern cpp_long [concrete = constants.%cpp_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc19_73.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.aec = addr_of %.loc19_73.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.2(%cpp_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_73.2: init %LongResult to %.loc19_73.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_73.3: ref %LongResult = temporary %.loc19_73.1, %.loc19_73.2
// CHECK:STDOUT:   %.loc19_73.4: %LongResult = acquire_value %.loc19_73.3
// CHECK:STDOUT:   %.loc19_34: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_result: %LongResult = wrapper_binding cpp_long_result, %.loc19_73.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.ae7 = value_binding_pattern cpp_long_result [concrete = constants.%cpp_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc21: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = wrapper_binding carbon_i64, %cpp_long_long.ref.loc21
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.f57 = value_binding_pattern carbon_i64 [concrete = constants.%carbon_i64.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_54: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_81.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.662 = addr_of %.loc22_81.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.1(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_81.2: init %LongLongResult to %.loc22_81.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_81.3: ref %LongLongResult = temporary %.loc22_81.1, %.loc22_81.2
// CHECK:STDOUT:   %.loc22_81.4: %LongLongResult = acquire_value %.loc22_81.3
// CHECK:STDOUT:   %.loc22_36: type = splice_block %LongLongResult.ref.loc22 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc22: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_i64_result: %LongLongResult = wrapper_binding carbon_i64_result, %.loc22_81.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.357 = value_binding_pattern carbon_i64_result [concrete = constants.%carbon_i64_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24_55: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn.loc24_55 [concrete = constants.%bound_method.3e4]
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
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.01b = impl_witness_access constants.%As.impl_witness.2e1, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d9]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.1: <specific function> = specific_function %impl.elem0.loc24_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %int_1.loc24, %specific_fn.loc24_30.1 [concrete = constants.%bound_method.991]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc24_30.2(%int_1.loc24) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc24_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc24_30.2: %i64 = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.779 = impl_witness_access constants.%ImplicitAs.impl_witness.a75, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.be5]
// CHECK:STDOUT:     %bound_method.loc24_30.3: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.2: <specific function> = specific_function %impl.elem0.loc24_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.4: <bound method> = bound_method %.loc24_30.2, %specific_fn.loc24_30.2 [concrete = constants.%bound_method.e21]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.4(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_81.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc22: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc22(%.loc22_81.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound: <bound method> = bound_method %.loc19_73.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call %LongResult.cpp_destructor.bound(%.loc19_73.3)
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
// CHECK:STDOUT: --- copy_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.418: <witness> = impl_witness imports.%Copy.impl_witness_table.30d [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.418) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.165: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a14: type = fn_type_with_self_type %Copy.WithSelf.Op.type.165, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.30d = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_28: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.a14 = impl_witness_access constants.%Copy.impl_witness.418, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_28, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc9(%.loc9_28)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete = constants.%b.var_patt]
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
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %EqWith.type.3ed: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.6db: <witness> = impl_witness imports.%EqWith.impl_witness_table.cf3 [concrete]
// CHECK:STDOUT:   %EqWith.facet.95b: %EqWith.type.3ed = facet_value %Cpp.long, (%EqWith.impl_witness.6db) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.1bc: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.95b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.6a3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.95b) [concrete]
// CHECK:STDOUT:   %.0e6: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.1bc, %EqWith.facet.95b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1ca: %Cpp.long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.f1b: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.6a3, %EqWith.facet.95b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a6a: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.484: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.3c9: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.47f [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.960: %OrderedWith.type.484 = facet_value %Cpp.long, (%OrderedWith.impl_witness.3c9) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b6e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.960) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.4a2: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.960) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.899: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.960) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.28b: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.960) [concrete]
// CHECK:STDOUT:   %.56d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.899, %OrderedWith.facet.960 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.7c8: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b6e, %OrderedWith.facet.960 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.606: %Cpp.long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.ac2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.28b, %OrderedWith.facet.960 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.d14: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.4a2, %OrderedWith.facet.960 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.a87: %Cpp.long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc9b: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.cf3 = impl_witness_table (%Core.import_ref.a87, %Core.import_ref.adc9b), @Cpp.long.as.EqWith.impl.f72 [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e8: %Cpp.long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757ec: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32c: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.47f = impl_witness_table (%Core.import_ref.0e8, %Core.import_ref.757ec, %Core.import_ref.c32c, %Core.import_ref.381), @Cpp.long.as.OrderedWith.impl.f51 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long = wrapper_binding a, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Cpp.long = wrapper_binding b, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ada = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.0e6 = impl_witness_access constants.%EqWith.impl_witness.6db, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.f1b = impl_witness_access constants.%EqWith.impl_witness.6db, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.56d = impl_witness_access constants.%OrderedWith.impl_witness.3c9, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.f43 = impl_witness_access constants.%OrderedWith.impl_witness.3c9, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.ac2 = impl_witness_access constants.%OrderedWith.impl_witness.3c9, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.d14 = impl_witness_access constants.%OrderedWith.impl_witness.3c9, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d67: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.1ae: %ImplicitAs.type.340 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.477445.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.367ca8.1: %Cpp.long.as.EqWith.impl.NotEqual.type.477445.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee5c37.1: %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.477445.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.367ca8.2: %Cpp.long.as.EqWith.impl.NotEqual.type.477445.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee5c37.2: %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %T.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6de1da.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.956(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.501deb.1: %T.as_type.as.EqWith.impl.NotEqual.type.6de1da.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.8af025.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.956(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c477d6.1: %T.as_type.as.EqWith.impl.Equal.type.8af025.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6de1da.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.956(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.501deb.2: %T.as_type.as.EqWith.impl.NotEqual.type.6de1da.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.8af025.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.956(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c477d6.2: %T.as_type.as.EqWith.impl.Equal.type.8af025.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.aee: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a3a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c71: %ImplicitAs.type.221 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.53b: <witness> = impl_witness imports.%EqWith.impl_witness_table.561, @T.as_type.as.EqWith.impl.956(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.9269d3.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.956(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c936a0.1: %T.as_type.as.EqWith.impl.Equal.type.9269d3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.956(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.44eea9.1: %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.9269d3.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.956(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c936a0.2: %T.as_type.as.EqWith.impl.Equal.type.9269d3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.956(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.44eea9.2: %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.768: %EqWith.type.d67 = facet_value %Cpp.long, (%EqWith.impl_witness.53b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.80c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.768) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.1dd: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.768) [concrete]
// CHECK:STDOUT:   %.da9: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.80c, %EqWith.facet.768 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.cba536.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c936a0.2, @T.as_type.as.EqWith.impl.Equal.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.cba536.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c936a0.1, @T.as_type.as.EqWith.impl.Equal.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.294: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.294, %ImplicitAs.facet.c71 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b06: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.1dd, %EqWith.facet.768 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.228325.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.44eea9.2, @T.as_type.as.EqWith.impl.NotEqual.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.228325.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.44eea9.1, @T.as_type.as.EqWith.impl.NotEqual.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.af1: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.029602.1: %Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.431fd2.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.458c1e.1: %Cpp.long.as.OrderedWith.impl.Less.type.431fd2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.029602.2: %Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.431fd2.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.d2f(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.458c1e.2: %Cpp.long.as.OrderedWith.impl.Less.type.431fd2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.772097.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.341bcf.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.249060.1: %T.as_type.as.OrderedWith.impl.Greater.type.341bcf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b513da.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.354913.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.0ec126.1: %T.as_type.as.OrderedWith.impl.Less.type.354913.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.772097.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.341bcf.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.249060.2: %T.as_type.as.OrderedWith.impl.Greater.type.341bcf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b513da.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.354913.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.7a0(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.0ec126.2: %T.as_type.as.OrderedWith.impl.Less.type.354913.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.975: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ee7: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.13e, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.de033b.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.8f6b78.1: %T.as_type.as.OrderedWith.impl.Less.type.de033b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.11de18.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.017256.1: %T.as_type.as.OrderedWith.impl.Greater.type.11de18.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.de033b.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.8f6b78.2: %T.as_type.as.OrderedWith.impl.Less.type.de033b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.11de18.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.017256.2: %T.as_type.as.OrderedWith.impl.Greater.type.11de18.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.7a0(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.226: %OrderedWith.type.af1 = facet_value %Cpp.long, (%OrderedWith.impl_witness.ee7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.643: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.226) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.025: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.226) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.7b6: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.226) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.436: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.226) [concrete]
// CHECK:STDOUT:   %.ffd: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.7b6, %OrderedWith.facet.226 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.17599d.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.017256.2, @T.as_type.as.OrderedWith.impl.Greater.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.17599d.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.017256.1, @T.as_type.as.OrderedWith.impl.Greater.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.da4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.643, %OrderedWith.facet.226 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.b43576.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.8f6b78.2, @T.as_type.as.OrderedWith.impl.Less.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.b43576.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.8f6b78.1, @T.as_type.as.OrderedWith.impl.Less.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.496: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.436, %OrderedWith.facet.226 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0e8d3e.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0e8d3e.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.fcb: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.025, %OrderedWith.facet.226 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.30cbb1.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.30cbb1.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.dcc: <witness> = impl_witness imports.%EqWith.impl_witness_table.dd7, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.76e71d.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.3b50c9.1: %Cpp.long.as.EqWith.impl.Equal.type.76e71d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.92f9dd.1: %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.76e71d.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.3b50c9.2: %Cpp.long.as.EqWith.impl.Equal.type.76e71d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.92f9dd.2: %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.f85: %EqWith.type.aee = facet_value %Cpp.long, (%EqWith.impl_witness.dcc) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.c8f: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.f85) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c7b: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.f85) [concrete]
// CHECK:STDOUT:   %.fe7: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.c8f, %EqWith.facet.f85 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f7b79e.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.3b50c9.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f7b79e.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.3b50c9.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.f7c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c7b, %EqWith.facet.f85 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.18e5a1.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.92f9dd.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.18e5a1.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.92f9dd.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.d99: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.afd, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.e47846.1: %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.803079.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.f69e67.1: %Cpp.long.as.OrderedWith.impl.Greater.type.803079.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.e47846.2: %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.803079.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.f69e67.2: %Cpp.long.as.OrderedWith.impl.Greater.type.803079.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.908: %OrderedWith.type.975 = facet_value %Cpp.long, (%OrderedWith.impl_witness.d99) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c5e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.908) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.735: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.908) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.e87: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.908) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.951: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.908) [concrete]
// CHECK:STDOUT:   %.7b2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.e87, %OrderedWith.facet.908 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.28a30d.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.f69e67.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.28a30d.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.f69e67.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.831: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c5e, %OrderedWith.facet.908 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.837d2f.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.e47846.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.837d2f.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.e47846.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.9b8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.951, %OrderedWith.facet.908 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.caf66c.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.caf66c.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.1b8: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.735, %OrderedWith.facet.908 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.cd0cbc.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.cd0cbc.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.1af: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.ee5c37.1)]
// CHECK:STDOUT:   %Core.import_ref.5b6: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.477445.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.367ca8.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.dd7 = impl_witness_table (%Core.import_ref.1af, %Core.import_ref.5b6), @Cpp.long.as.EqWith.impl.e2d [concrete]
// CHECK:STDOUT:   %Core.Equal.94f: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.ee5c37.2)]
// CHECK:STDOUT:   %Core.NotEqual.3eb: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.477445.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.367ca8.2)]
// CHECK:STDOUT:   %Core.import_ref.a9a: @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.8af025.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.c477d6.1)]
// CHECK:STDOUT:   %Core.import_ref.76a: @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.6de1da.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.501deb.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.561 = impl_witness_table (%Core.import_ref.a9a, %Core.import_ref.76a), @T.as_type.as.EqWith.impl.956 [concrete]
// CHECK:STDOUT:   %Core.Equal.970: @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.8af025.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.c477d6.2)]
// CHECK:STDOUT:   %Core.NotEqual.228: @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.6de1da.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.956.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.501deb.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a3a = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.c8d [concrete]
// CHECK:STDOUT:   %Core.import_ref.49e: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.431fd2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.458c1e.1)]
// CHECK:STDOUT:   %Core.import_ref.5ef: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.1)]
// CHECK:STDOUT:   %Core.import_ref.3459: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.029602.1)]
// CHECK:STDOUT:   %Core.import_ref.ccd: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.afd = impl_witness_table (%Core.import_ref.49e, %Core.import_ref.5ef, %Core.import_ref.3459, %Core.import_ref.ccd), @Cpp.long.as.OrderedWith.impl.d2f [concrete]
// CHECK:STDOUT:   %Core.Less.890: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.431fd2.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.458c1e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.a1b: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.2)]
// CHECK:STDOUT:   %Core.Greater.bd4: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.029602.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.829: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.2)]
// CHECK:STDOUT:   %Core.import_ref.b96: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.354913.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.0ec126.1)]
// CHECK:STDOUT:   %Core.import_ref.870: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.b513da.1)]
// CHECK:STDOUT:   %Core.import_ref.879: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.341bcf.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.249060.1)]
// CHECK:STDOUT:   %Core.import_ref.9a7: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.772097.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.13e = impl_witness_table (%Core.import_ref.b96, %Core.import_ref.870, %Core.import_ref.879, %Core.import_ref.9a7), @T.as_type.as.OrderedWith.impl.7a0 [concrete]
// CHECK:STDOUT:   %Core.Less.ccb: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.354913.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.0ec126.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.cbb: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.9b3e8c.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.b513da.2)]
// CHECK:STDOUT:   %Core.Greater.a52: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.341bcf.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.249060.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.a3d: @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.e45fd7.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.7a0.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.772097.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long = wrapper_binding a, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.da9 = impl_witness_access constants.%EqWith.impl_witness.53b, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c936a0.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.cba536.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.9269d3.1 = specific_constant imports.%Core.Equal.970, @T.as_type.as.EqWith.impl.956(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c936a0.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.9269d3.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c936a0.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.cba536.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_3(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %i64 = converted %a.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.b06 = impl_witness_access constants.%EqWith.impl_witness.53b, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.44eea9.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.228325.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.1 = specific_constant imports.%Core.NotEqual.228, @T.as_type.as.EqWith.impl.956(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.44eea9.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.8585e2.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.44eea9.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.228325.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_3(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %i64 = converted %a.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.ffd = impl_witness_access constants.%OrderedWith.impl_witness.ee7, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.017256.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.17599d.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.11de18.1 = specific_constant imports.%Core.Greater.a52, @T.as_type.as.OrderedWith.impl.7a0(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.017256.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.11de18.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.017256.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.17599d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_3(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %i64 = converted %a.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.da4 = impl_witness_access constants.%OrderedWith.impl_witness.ee7, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.8f6b78.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.b43576.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.de033b.1 = specific_constant imports.%Core.Less.ccb, @T.as_type.as.OrderedWith.impl.7a0(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.8f6b78.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.de033b.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.8f6b78.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.b43576.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_3(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %i64 = converted %a.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.496 = impl_witness_access constants.%OrderedWith.impl_witness.ee7, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0e8d3e.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.1 = specific_constant imports.%Core.GreaterOrEquivalent.a3d, @T.as_type.as.OrderedWith.impl.7a0(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1eda94.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6f44e0.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0e8d3e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_3(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %i64 = converted %a.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.fcb = impl_witness_access constants.%OrderedWith.impl_witness.ee7, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.30cbb1.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.1 = specific_constant imports.%Core.LessOrEquivalent.cbb, @T.as_type.as.OrderedWith.impl.7a0(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.75d39c.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.d36d70.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.30cbb1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_3(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %i64 = converted %a.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %b.ref.loc15)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.fe7 = impl_witness_access constants.%EqWith.impl_witness.dcc, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3b50c9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f7b79e.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.EqWith.impl.Equal.type.76e71d.1 = specific_constant imports.%Core.Equal.94f, @Cpp.long.as.EqWith.impl.e2d(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3b50c9.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.76e71d.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3b50c9.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f7b79e.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.f7c = impl_witness_access constants.%EqWith.impl_witness.dcc, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.92f9dd.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.18e5a1.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.1 = specific_constant imports.%Core.NotEqual.3eb, @Cpp.long.as.EqWith.impl.e2d(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.92f9dd.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.9fc84a.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.92f9dd.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.18e5a1.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.7b2 = impl_witness_access constants.%OrderedWith.impl_witness.d99, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.f69e67.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.28a30d.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.803079.1 = specific_constant imports.%Core.Greater.bd4, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.f69e67.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.803079.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.f69e67.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.28a30d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.831 = impl_witness_access constants.%OrderedWith.impl_witness.d99, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.e47846.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.837d2f.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.1 = specific_constant imports.%Core.Less.890, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.e47846.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.e8e281.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.e47846.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.837d2f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.9b8 = impl_witness_access constants.%OrderedWith.impl_witness.d99, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.caf66c.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.1 = specific_constant imports.%Core.GreaterOrEquivalent.829, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.6a2e90.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.6badd7.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.caf66c.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc21: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.1b8 = impl_witness_access constants.%OrderedWith.impl_witness.d99, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.cd0cbc.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.1 = specific_constant imports.%Core.LessOrEquivalent.a1b, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.07caf1.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1d76f6.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.210) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.cd0cbc.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc22: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc22_5.3(%a.ref.loc22, %.loc22_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.3ed: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.1ae: %ImplicitAs.type.340 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.4349fe.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.7e7(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.50a028.1: %T.as_type.as.EqWith.impl.NotEqual.type.4349fe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f384d8.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.7e7(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.99b5a3.1: %T.as_type.as.EqWith.impl.Equal.type.f384d8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.4349fe.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.7e7(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.50a028.2: %T.as_type.as.EqWith.impl.NotEqual.type.4349fe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f384d8.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.7e7(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.99b5a3.2: %T.as_type.as.EqWith.impl.Equal.type.f384d8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %T.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.97ec42.1: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.2b8(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.c6c584.1: %Int.as.EqWith.impl.NotEqual.type.97ec42.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.c9381b.1: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.2b8(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.a4cb34.1: %Int.as.EqWith.impl.Equal.type.c9381b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.97ec42.2: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.2b8(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.c6c584.2: %Int.as.EqWith.impl.NotEqual.type.97ec42.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.c9381b.2: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.2b8(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.a4cb34.2: %Int.as.EqWith.impl.Equal.type.c9381b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a3a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c71: %ImplicitAs.type.221 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.c6b: <witness> = impl_witness imports.%EqWith.impl_witness_table.d4a, @Int.as.EqWith.impl.2b8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.5ee80b.1: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.2b8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.6fef86.1: %Int.as.EqWith.impl.Equal.type.5ee80b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.3ca782.1: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.2b8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.c74654.1: %Int.as.EqWith.impl.NotEqual.type.3ca782.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.5ee80b.2: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.2b8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.6fef86.2: %Int.as.EqWith.impl.Equal.type.5ee80b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.3ca782.2: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.2b8(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.c74654.2: %Int.as.EqWith.impl.NotEqual.type.3ca782.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.685: %EqWith.type.3ed = facet_value %i64, (%EqWith.impl_witness.c6b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.83c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.685) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.dc2: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.685) [concrete]
// CHECK:STDOUT:   %.b68: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.83c, %EqWith.facet.685 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.1d8cd8.1: <specific function> = specific_function %Int.as.EqWith.impl.Equal.6fef86.2, @Int.as.EqWith.impl.Equal.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.1d8cd8.2: <specific function> = specific_function %Int.as.EqWith.impl.Equal.6fef86.1, @Int.as.EqWith.impl.Equal.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.294: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.294, %ImplicitAs.facet.c71 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.864: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.dc2, %EqWith.facet.685 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.f14bf7.1: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.c74654.2, @Int.as.EqWith.impl.NotEqual.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.f14bf7.2: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.c74654.1, @Int.as.EqWith.impl.NotEqual.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.484: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.776eb1.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bf137f.1: %T.as_type.as.OrderedWith.impl.Greater.type.776eb1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.9ca225.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.571834.1: %T.as_type.as.OrderedWith.impl.Less.type.9ca225.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.776eb1.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bf137f.2: %T.as_type.as.OrderedWith.impl.Greater.type.776eb1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.9ca225.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.4ed(%T.1ae) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.571834.2: %T.as_type.as.OrderedWith.impl.Less.type.9ca225.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.1c7f7d.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.18c27f.1: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.cbccb6.1: %Int.as.OrderedWith.impl.Greater.type.18c27f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.7bb73f.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.6605e2.1: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.59643b.1: %Int.as.OrderedWith.impl.Less.type.6605e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.1c7f7d.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.18c27f.2: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.cbccb6.2: %Int.as.OrderedWith.impl.Greater.type.18c27f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.7bb73f.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.6605e2.2: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.f36(%N, %T.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.59643b.2: %Int.as.OrderedWith.impl.Less.type.6605e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.a98: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.b1a, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.7135f0.1: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.1c6761.1: %Int.as.OrderedWith.impl.Less.type.7135f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.621392.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.d750f4.1: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.0ec18c.1: %Int.as.OrderedWith.impl.Greater.type.d750f4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.7135f0.2: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.1c6761.2: %Int.as.OrderedWith.impl.Less.type.7135f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.621392.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.d750f4.2: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.0ec18c.2: %Int.as.OrderedWith.impl.Greater.type.d750f4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.f36(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.c3a: %OrderedWith.type.484 = facet_value %i64, (%OrderedWith.impl_witness.a98) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.55f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.c3a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.d47: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.c3a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.400: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.c3a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a77: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.c3a) [concrete]
// CHECK:STDOUT:   %.c7f: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.400, %OrderedWith.facet.c3a [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.980c5e.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.0ec18c.2, @Int.as.OrderedWith.impl.Greater.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.980c5e.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.0ec18c.1, @Int.as.OrderedWith.impl.Greater.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.e4f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.55f, %OrderedWith.facet.c3a [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.1c1cdd.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.1c6761.2, @Int.as.OrderedWith.impl.Less.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.1c1cdd.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.1c6761.1, @Int.as.OrderedWith.impl.Less.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.ce8f: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a77, %OrderedWith.facet.c3a [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2fed0f.1: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.2, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2fed0f.2: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.1, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.2df: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.d47, %OrderedWith.facet.c3a [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3843b4.1: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.621392.2, @Int.as.OrderedWith.impl.LessOrEquivalent.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3843b4.2: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.621392.1, @Int.as.OrderedWith.impl.LessOrEquivalent.3(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.b0f: <witness> = impl_witness imports.%EqWith.impl_witness_table.195, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.e098d7.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.32ee3c.1: %T.as_type.as.EqWith.impl.Equal.type.e098d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.992176.1: %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.e098d7.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.32ee3c.2: %T.as_type.as.EqWith.impl.Equal.type.e098d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.992176.2: %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.bf7: %EqWith.type.3ed = facet_value Core.IntLiteral, (%EqWith.impl_witness.b0f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.1e5: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.bf7) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.bd5: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.bf7) [concrete]
// CHECK:STDOUT:   %.be4: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.1e5, %EqWith.facet.bf7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.c484ca.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.32ee3c.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.32ee3c.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.573be8.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.c484ca.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.32ee3c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.32ee3c.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.573be8.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.2 [concrete]
// CHECK:STDOUT:   %.943: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.bd5, %EqWith.facet.bf7 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.744760.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.992176.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.992176.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.bd39a5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.744760.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.992176.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.992176.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.bd39a5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.87d: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.f65, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.7fdd0f.1: %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a7d138.1: %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.7fdd0f.2: %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a7d138.2: %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.001: %OrderedWith.type.484 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.87d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.637: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.001) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3fb: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.001) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.992: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.001) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9de: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.001) [concrete]
// CHECK:STDOUT:   %.4ce: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.992, %OrderedWith.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.c08816.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.a7d138.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a7d138.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.a6774b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.c08816.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.a7d138.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a7d138.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.a6774b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.2 [concrete]
// CHECK:STDOUT:   %.90a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.637, %OrderedWith.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.df13e5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.7fdd0f.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.7fdd0f.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.65c00c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.df13e5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.7fdd0f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.7fdd0f.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.65c00c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.2 [concrete]
// CHECK:STDOUT:   %.70d: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9de, %OrderedWith.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4bee52.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.0f34a9.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4bee52.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.0f34a9.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.2 [concrete]
// CHECK:STDOUT:   %.f0a: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3fb, %OrderedWith.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.931d3c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.cc0313.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.931d3c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %bound_method.cc0313.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.ad0: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.f384d8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.99b5a3.1)]
// CHECK:STDOUT:   %Core.import_ref.8c0: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.4349fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.50a028.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.195 = impl_witness_table (%Core.import_ref.ad0, %Core.import_ref.8c0), @T.as_type.as.EqWith.impl.7e7 [concrete]
// CHECK:STDOUT:   %Core.Equal.c5f: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.f384d8.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.99b5a3.2)]
// CHECK:STDOUT:   %Core.NotEqual.e58: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.4349fe.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.50a028.2)]
// CHECK:STDOUT:   %Core.import_ref.60d: @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.Equal.type.2 (%Int.as.EqWith.impl.Equal.type.c9381b.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.Equal.2 (constants.%Int.as.EqWith.impl.Equal.a4cb34.1)]
// CHECK:STDOUT:   %Core.import_ref.fb5: @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.NotEqual.type.2 (%Int.as.EqWith.impl.NotEqual.type.97ec42.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.NotEqual.2 (constants.%Int.as.EqWith.impl.NotEqual.c6c584.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.d4a = impl_witness_table (%Core.import_ref.60d, %Core.import_ref.fb5), @Int.as.EqWith.impl.2b8 [concrete]
// CHECK:STDOUT:   %Core.Equal.926: @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.Equal.type.1 (%Int.as.EqWith.impl.Equal.type.c9381b.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.Equal.1 (constants.%Int.as.EqWith.impl.Equal.a4cb34.2)]
// CHECK:STDOUT:   %Core.NotEqual.c83: @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.NotEqual.type.1 (%Int.as.EqWith.impl.NotEqual.type.97ec42.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @Int.as.EqWith.impl.2b8.%Int.as.EqWith.impl.NotEqual.1 (constants.%Int.as.EqWith.impl.NotEqual.c6c584.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a3a = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.c8d [concrete]
// CHECK:STDOUT:   %Core.import_ref.011: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.9ca225.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.571834.1)]
// CHECK:STDOUT:   %Core.import_ref.ba20: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.1)]
// CHECK:STDOUT:   %Core.import_ref.44d: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.776eb1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.bf137f.1)]
// CHECK:STDOUT:   %Core.import_ref.3aa: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.f65 = impl_witness_table (%Core.import_ref.011, %Core.import_ref.ba20, %Core.import_ref.44d, %Core.import_ref.3aa), @T.as_type.as.OrderedWith.impl.4ed [concrete]
// CHECK:STDOUT:   %Core.Less.1d9: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.9ca225.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.571834.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.88c: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.2)]
// CHECK:STDOUT:   %Core.Greater.172: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.776eb1.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.bf137f.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.09a: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.2)]
// CHECK:STDOUT:   %Core.import_ref.1dc: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Less.type.2 (%Int.as.OrderedWith.impl.Less.type.6605e2.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Less.2 (constants.%Int.as.OrderedWith.impl.Less.59643b.1)]
// CHECK:STDOUT:   %Core.import_ref.2a5: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.7bb73f.1)]
// CHECK:STDOUT:   %Core.import_ref.0ab: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Greater.type.2 (%Int.as.OrderedWith.impl.Greater.type.18c27f.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Greater.2 (constants.%Int.as.OrderedWith.impl.Greater.cbccb6.1)]
// CHECK:STDOUT:   %Core.import_ref.a4c6: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1c7f7d.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.b1a = impl_witness_table (%Core.import_ref.1dc, %Core.import_ref.2a5, %Core.import_ref.0ab, %Core.import_ref.a4c6), @Int.as.OrderedWith.impl.f36 [concrete]
// CHECK:STDOUT:   %Core.Less.80e: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Less.type.1 (%Int.as.OrderedWith.impl.Less.type.6605e2.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Less.1 (constants.%Int.as.OrderedWith.impl.Less.59643b.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.633: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.0ff510.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.7bb73f.2)]
// CHECK:STDOUT:   %Core.Greater.28d: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Greater.type.1 (%Int.as.OrderedWith.impl.Greater.type.18c27f.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.Greater.1 (constants.%Int.as.OrderedWith.impl.Greater.cbccb6.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.b3a: @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.c3cac1.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.f36.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1c7f7d.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Cpp.long = wrapper_binding a, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.b68 = impl_witness_access constants.%EqWith.impl_witness.c6b, element0 [concrete = constants.%Int.as.EqWith.impl.Equal.6fef86.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.EqWith.impl.Equal.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.1d8cd8.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.EqWith.impl.Equal.type.5ee80b.1 = specific_constant imports.%Core.Equal.926, @Int.as.EqWith.impl.2b8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.Equal.6fef86.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Int.as.EqWith.impl.Equal.type.5ee80b.1 = name_ref Equal, %.loc10_5 [concrete = constants.%Int.as.EqWith.impl.Equal.6fef86.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @Int.as.EqWith.impl.Equal.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.1d8cd8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %Int.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %a.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%b.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.864 = impl_witness_access constants.%EqWith.impl_witness.c6b, element1 [concrete = constants.%Int.as.EqWith.impl.NotEqual.c74654.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Int.as.EqWith.impl.NotEqual.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.f14bf7.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.EqWith.impl.NotEqual.type.3ca782.1 = specific_constant imports.%Core.NotEqual.c83, @Int.as.EqWith.impl.2b8(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.NotEqual.c74654.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Int.as.EqWith.impl.NotEqual.type.3ca782.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%Int.as.EqWith.impl.NotEqual.c74654.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @Int.as.EqWith.impl.NotEqual.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.f14bf7.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %Int.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %a.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%b.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.c7f = impl_witness_access constants.%OrderedWith.impl_witness.a98, element2 [concrete = constants.%Int.as.OrderedWith.impl.Greater.0ec18c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Int.as.OrderedWith.impl.Greater.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.980c5e.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.OrderedWith.impl.Greater.type.d750f4.1 = specific_constant imports.%Core.Greater.28d, @Int.as.OrderedWith.impl.f36(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Greater.0ec18c.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Int.as.OrderedWith.impl.Greater.type.d750f4.1 = name_ref Greater, %.loc12_5 [concrete = constants.%Int.as.OrderedWith.impl.Greater.0ec18c.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @Int.as.OrderedWith.impl.Greater.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.980c5e.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %Int.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_7(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %i64 = converted %a.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%b.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.e4f = impl_witness_access constants.%OrderedWith.impl_witness.a98, element0 [concrete = constants.%Int.as.OrderedWith.impl.Less.1c6761.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.OrderedWith.impl.Less.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.1c1cdd.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.OrderedWith.impl.Less.type.7135f0.1 = specific_constant imports.%Core.Less.80e, @Int.as.OrderedWith.impl.f36(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Less.1c6761.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Int.as.OrderedWith.impl.Less.type.7135f0.1 = name_ref Less, %.loc13_5 [concrete = constants.%Int.as.OrderedWith.impl.Less.1c6761.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @Int.as.OrderedWith.impl.Less.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.1c1cdd.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %Int.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_7(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %i64 = converted %a.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%b.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.ce8f = impl_witness_access constants.%OrderedWith.impl_witness.a98, element3 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2fed0f.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.1 = specific_constant imports.%Core.GreaterOrEquivalent.b3a, @Int.as.OrderedWith.impl.f36(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.5bb690.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.ca4fe3.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2fed0f.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %a.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%b.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.2df = impl_witness_access constants.%OrderedWith.impl_witness.a98, element1 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.621392.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3843b4.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.1 = specific_constant imports.%Core.LessOrEquivalent.633, @Int.as.OrderedWith.impl.f36(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.621392.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Int.as.OrderedWith.impl.LessOrEquivalent.type.07dffd.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.621392.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3843b4.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_8(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %i64 = converted %a.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%b.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.be4 = impl_witness_access constants.%EqWith.impl_witness.b0f, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.32ee3c.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.c484ca.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.573be8.1]
// CHECK:STDOUT:   %.loc17_5.3: %T.as_type.as.EqWith.impl.Equal.type.e098d7.1 = specific_constant imports.%Core.Equal.c5f, @T.as_type.as.EqWith.impl.7e7(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.32ee3c.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.e098d7.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.32ee3c.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.c484ca.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.90c3dc.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn [concrete = constants.%bound_method.573be8.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.943 = impl_witness_access constants.%EqWith.impl_witness.b0f, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.992176.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.744760.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.bd39a5.1]
// CHECK:STDOUT:   %.loc18_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.1 = specific_constant imports.%Core.NotEqual.e58, @T.as_type.as.EqWith.impl.7e7(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.992176.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.2267b3.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.992176.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.744760.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.fabde5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn [concrete = constants.%bound_method.bd39a5.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.4ce = impl_witness_access constants.%OrderedWith.impl_witness.87d, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a7d138.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.c08816.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.a6774b.1]
// CHECK:STDOUT:   %.loc19_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.1 = specific_constant imports.%Core.Greater.172, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a7d138.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.5b267a.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a7d138.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.c08816.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.a03db4.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn [concrete = constants.%bound_method.a6774b.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.90a = impl_witness_access constants.%OrderedWith.impl_witness.87d, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7fdd0f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.df13e5.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.65c00c.1]
// CHECK:STDOUT:   %.loc20_5.3: %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.1 = specific_constant imports.%Core.Less.1d9, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7fdd0f.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.8fcddd.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.7fdd0f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.df13e5.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.7a5c67.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn [concrete = constants.%bound_method.65c00c.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.70d = impl_witness_access constants.%OrderedWith.impl_witness.87d, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4bee52.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.0f34a9.1]
// CHECK:STDOUT:   %.loc21_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.1 = specific_constant imports.%Core.GreaterOrEquivalent.09a, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b542aa.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8990d1.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.4bee52.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.6be80e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn [concrete = constants.%bound_method.0f34a9.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.f0a = impl_witness_access constants.%OrderedWith.impl_witness.87d, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.931d3c.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.210]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.cc0313.1]
// CHECK:STDOUT:   %.loc22_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.1 = specific_constant imports.%Core.LessOrEquivalent.88c, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.a14e77.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.6eace1.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.931d3c.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.210) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19ea46.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn [concrete = constants.%bound_method.cc0313.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.6c4: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.61d: %T.as_type.as.ImplicitAs.impl.Convert.type.6c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e06: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.1ae: %ImplicitAs.type.340 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.418: <witness> = impl_witness imports.%Copy.impl_witness_table.30d [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.418) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b9, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4c3: %ImplicitAs.type.340 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.0a0) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.f93: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.338, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.6dc032.1: %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.6dc032.2: %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.bbe: %AddAssignWith.type.e06 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.f93) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.8ff: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.bbe) [concrete]
// CHECK:STDOUT:   %.8a7: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.8ff, %AddAssignWith.facet.bbe [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9e8d09.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.6dc032.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9e8d09.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.6dc032.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.6ba: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c163a1.1: %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c163a1.2: %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.5f8: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.219, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.1: %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.2: %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.038: %SubAssignWith.type.6ba = facet_value %Cpp.long, (%SubAssignWith.impl_witness.5f8) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.065: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.038) [concrete]
// CHECK:STDOUT:   %.52a: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.065, %SubAssignWith.facet.038 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.95343b.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.95343b.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.642: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5870c9.1: %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5870c9.2: %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.2df: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.7c3, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.1: %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.2: %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.898: %MulAssignWith.type.642 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.2df) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.587: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.898) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.587, %MulAssignWith.facet.898 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.f4a246.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.f4a246.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.64b: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.302d61.1: %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.302d61.2: %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.47c: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.c1b, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.2080ca.1: %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.2080ca.2: %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.e69: %DivAssignWith.type.64b = facet_value %Cpp.long, (%DivAssignWith.impl_witness.47c) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.141: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.e69) [concrete]
// CHECK:STDOUT:   %.76b: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.141, %DivAssignWith.facet.e69 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ea3d38.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2080ca.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ea3d38.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2080ca.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.132: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.d0e55c.1: %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.d0e55c.2: %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.651: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.fc0, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.1: %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.2: %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.428: %ModAssignWith.type.132 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.651) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.912: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.428) [concrete]
// CHECK:STDOUT:   %.b68: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.912, %ModAssignWith.facet.428 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.0903fb.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.0903fb.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.44e: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.999389.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.999389.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.2eb: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.966, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.257: %BitAndAssignWith.type.44e = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.2eb) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.168: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.257) [concrete]
// CHECK:STDOUT:   %.48c: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.168, %BitAndAssignWith.facet.257 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.6f1afe.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.6f1afe.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.e7d: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.0bb: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.22c, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.fa5: %BitOrAssignWith.type.e7d = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.0bb) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.fd3: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.fa5) [concrete]
// CHECK:STDOUT:   %.4cc: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.fd3, %BitOrAssignWith.facet.fa5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8bbade.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8bbade.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ce1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.2c6: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.98a, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.28a: %BitXorAssignWith.type.ce1 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.2c6) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.9b8: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.28a) [concrete]
// CHECK:STDOUT:   %.569: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.9b8, %BitXorAssignWith.facet.28a [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.f62acd.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.f62acd.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.452: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.d34: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.041, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.d3d: %LeftShiftAssignWith.type.452 = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.d34) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.9a1: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.d3d) [concrete]
// CHECK:STDOUT:   %.cf6: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.9a1, %LeftShiftAssignWith.facet.d3d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.34c23f.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.34c23f.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f17: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.350: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.d21, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.679: %RightShiftAssignWith.type.f17 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.350) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.bcc: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.679) [concrete]
// CHECK:STDOUT:   %.2db: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.bcc, %RightShiftAssignWith.facet.679 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.ac9b9d.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.ac9b9d.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.4c3) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.051: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.6c4) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.61d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b9 = impl_witness_table (%Core.import_ref.051), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.d3c: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.338 = impl_witness_table (%Core.import_ref.d3c), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.373: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.30d = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f7887: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c163a1.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.219 = impl_witness_table (%Core.import_ref.f7887), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.789: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c163a1.2)]
// CHECK:STDOUT:   %Core.import_ref.094: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.5870c9.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.7c3 = impl_witness_table (%Core.import_ref.094), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b2c: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.5870c9.2)]
// CHECK:STDOUT:   %Core.import_ref.a6a: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.302d61.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.c1b = impl_witness_table (%Core.import_ref.a6a), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f8d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.302d61.2)]
// CHECK:STDOUT:   %Core.import_ref.56bc: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.d0e55c.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.fc0 = impl_witness_table (%Core.import_ref.56bc), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6f1: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.d0e55c.2)]
// CHECK:STDOUT:   %Core.import_ref.130: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.999389.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.966 = impl_witness_table (%Core.import_ref.130), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.09f: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.999389.2)]
// CHECK:STDOUT:   %Core.import_ref.839: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.22c = impl_witness_table (%Core.import_ref.839), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.607: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.2)]
// CHECK:STDOUT:   %Core.import_ref.39ee: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.98a = impl_witness_table (%Core.import_ref.39ee), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aed: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.2)]
// CHECK:STDOUT:   %Core.import_ref.7f9: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.041 = impl_witness_table (%Core.import_ref.7f9), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.114: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.2)]
// CHECK:STDOUT:   %Core.import_ref.919: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.d21 = impl_witness_table (%Core.import_ref.919), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.592: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var_storage %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.8a7 = impl_witness_access constants.%AddAssignWith.impl_witness.f93, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.6dc032.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9e8d09.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.1 = specific_constant imports.%Core.Op.373, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.6dc032.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.3d63eb.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.6dc032.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9e8d09.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.52a = impl_witness_access constants.%SubAssignWith.impl_witness.5f8, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.95343b.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.1 = specific_constant imports.%Core.Op.789, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.57341e.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.a0a7a6.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.95343b.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.7e9 = impl_witness_access constants.%MulAssignWith.impl_witness.2df, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.f4a246.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.1 = specific_constant imports.%Core.Op.b2c, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.f7a36d.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4a1a0a.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.f4a246.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.76b = impl_witness_access constants.%DivAssignWith.impl_witness.47c, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2080ca.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ea3d38.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.1 = specific_constant imports.%Core.Op.f8d, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2080ca.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.00309f.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2080ca.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ea3d38.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.b68 = impl_witness_access constants.%ModAssignWith.impl_witness.651, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.0903fb.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.1 = specific_constant imports.%Core.Op.6f1, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.f1e8d7.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c3c3e1.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.0903fb.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.48c = impl_witness_access constants.%BitAndAssignWith.impl_witness.2eb, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.6f1afe.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.1 = specific_constant imports.%Core.Op.09f, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.810a37.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.ee7c77.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.6f1afe.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.4cc = impl_witness_access constants.%BitOrAssignWith.impl_witness.0bb, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8bbade.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.1 = specific_constant imports.%Core.Op.607, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64caa1.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0dfa3a.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8bbade.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.569 = impl_witness_access constants.%BitXorAssignWith.impl_witness.2c6, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.f62acd.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.1 = specific_constant imports.%Core.Op.aed, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.1a1f6e.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.4e19d3.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.f62acd.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.cf6 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.d34, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.34c23f.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.1 = specific_constant imports.%Core.Op.114, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ec8ae6.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c3371c.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.34c23f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.2db = impl_witness_access constants.%RightShiftAssignWith.impl_witness.350, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0a0) [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.4c3]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.ac9b9d.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.1 = specific_constant imports.%Core.Op.592, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.5465ae.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d03ce7.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.4c3) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.ac9b9d.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
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
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e06: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a3a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c71: %ImplicitAs.type.221 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3ce: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.c0b, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.d4b879.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.4484b8.1: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.d4b879.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.4484b8.2: %Int.as.AddAssignWith.impl.Op.type.d4b879.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.2ea: %AddAssignWith.type.e06 = facet_value %i64, (%AddAssignWith.impl_witness.3ce) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.987: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.f16: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.987, %AddAssignWith.facet.2ea [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.4484b8.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.4484b8.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.294: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.294, %ImplicitAs.facet.c71 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.6ba: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.8ef325.1: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.f183ca.1: %Int.as.SubAssignWith.impl.Op.type.8ef325.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.8ef325.2: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.f183ca.2: %Int.as.SubAssignWith.impl.Op.type.8ef325.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.9f7: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.ed9, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.a048ce.1: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.880561.1: %Int.as.SubAssignWith.impl.Op.type.a048ce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.a048ce.2: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.880561.2: %Int.as.SubAssignWith.impl.Op.type.a048ce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.ae4: %SubAssignWith.type.6ba = facet_value %i64, (%SubAssignWith.impl_witness.9f7) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.bb2: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.ae4) [concrete]
// CHECK:STDOUT:   %.d91: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.bb2, %SubAssignWith.facet.ae4 [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.854c3f.1: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.880561.2, @Int.as.SubAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.854c3f.2: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.880561.1, @Int.as.SubAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.642: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.99e92a.1: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.1fe857.1: %Int.as.MulAssignWith.impl.Op.type.99e92a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.99e92a.2: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.1fe857.2: %Int.as.MulAssignWith.impl.Op.type.99e92a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.5a7: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.bb2, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.25fd2b.1: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.700938.1: %Int.as.MulAssignWith.impl.Op.type.25fd2b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.25fd2b.2: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.700938.2: %Int.as.MulAssignWith.impl.Op.type.25fd2b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.203: %MulAssignWith.type.642 = facet_value %i64, (%MulAssignWith.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.331: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.203) [concrete]
// CHECK:STDOUT:   %.7b5: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.331, %MulAssignWith.facet.203 [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.e90d66.1: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.700938.2, @Int.as.MulAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.e90d66.2: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.700938.1, @Int.as.MulAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.64b: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.33a652.1: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.e8972a.1: %Int.as.DivAssignWith.impl.Op.type.33a652.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.33a652.2: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.e8972a.2: %Int.as.DivAssignWith.impl.Op.type.33a652.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.39f: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.ff8, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.92ce4f.1: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.c6930d.1: %Int.as.DivAssignWith.impl.Op.type.92ce4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.92ce4f.2: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.c6930d.2: %Int.as.DivAssignWith.impl.Op.type.92ce4f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.4a4: %DivAssignWith.type.64b = facet_value %i64, (%DivAssignWith.impl_witness.39f) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.f0d: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.4a4) [concrete]
// CHECK:STDOUT:   %.c68: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.f0d, %DivAssignWith.facet.4a4 [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.0454de.1: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.c6930d.2, @Int.as.DivAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.0454de.2: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.c6930d.1, @Int.as.DivAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.132: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.42e78c.1: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.12511c.1: %Int.as.ModAssignWith.impl.Op.type.42e78c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.42e78c.2: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.12511c.2: %Int.as.ModAssignWith.impl.Op.type.42e78c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.a1c: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.1e5, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.9b6f30.1: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.b03bdb.1: %Int.as.ModAssignWith.impl.Op.type.9b6f30.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.9b6f30.2: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.b03bdb.2: %Int.as.ModAssignWith.impl.Op.type.9b6f30.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.591: %ModAssignWith.type.132 = facet_value %i64, (%ModAssignWith.impl_witness.a1c) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.bd3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.591) [concrete]
// CHECK:STDOUT:   %.360: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.bd3, %ModAssignWith.facet.591 [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.143374.1: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.b03bdb.2, @Int.as.ModAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.143374.2: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.b03bdb.1, @Int.as.ModAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.44e: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.490e73.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.c3b0c6.1: %Int.as.BitAndAssignWith.impl.Op.type.490e73.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.490e73.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.c3b0c6.2: %Int.as.BitAndAssignWith.impl.Op.type.490e73.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.2c7: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.b95, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.bbaece.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.096e1a.1: %Int.as.BitAndAssignWith.impl.Op.type.bbaece.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.bbaece.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.096e1a.2: %Int.as.BitAndAssignWith.impl.Op.type.bbaece.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.b9a: %BitAndAssignWith.type.44e = facet_value %i64, (%BitAndAssignWith.impl_witness.2c7) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.fe9: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.b9a) [concrete]
// CHECK:STDOUT:   %.706: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.fe9, %BitAndAssignWith.facet.b9a [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.4f006b.1: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.096e1a.2, @Int.as.BitAndAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.4f006b.2: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.096e1a.1, @Int.as.BitAndAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.e7d: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.f57ae9.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.d3cc0c.1: %Int.as.BitOrAssignWith.impl.Op.type.f57ae9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.f57ae9.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.d3cc0c.2: %Int.as.BitOrAssignWith.impl.Op.type.f57ae9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.084: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.80a, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.9c3899.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.508e1b.1: %Int.as.BitOrAssignWith.impl.Op.type.9c3899.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.9c3899.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.508e1b.2: %Int.as.BitOrAssignWith.impl.Op.type.9c3899.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.db7: %BitOrAssignWith.type.e7d = facet_value %i64, (%BitOrAssignWith.impl_witness.084) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.a8d: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.db7) [concrete]
// CHECK:STDOUT:   %.aa5: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.a8d, %BitOrAssignWith.facet.db7 [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.67f0b3.1: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.508e1b.2, @Int.as.BitOrAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.67f0b3.2: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.508e1b.1, @Int.as.BitOrAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ce1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.6fa003.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.492b4e.1: %Int.as.BitXorAssignWith.impl.Op.type.6fa003.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.6fa003.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.492b4e.2: %Int.as.BitXorAssignWith.impl.Op.type.6fa003.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.8bd: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.ed2, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.1a4108.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.e58813.1: %Int.as.BitXorAssignWith.impl.Op.type.1a4108.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.1a4108.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.e58813.2: %Int.as.BitXorAssignWith.impl.Op.type.1a4108.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.c03: %BitXorAssignWith.type.ce1 = facet_value %i64, (%BitXorAssignWith.impl_witness.8bd) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.426: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.c03) [concrete]
// CHECK:STDOUT:   %.6a6: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.426, %BitXorAssignWith.facet.c03 [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.462aaf.1: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.e58813.2, @Int.as.BitXorAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.462aaf.2: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.e58813.1, @Int.as.BitXorAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.452: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.d04a1c.1: %Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.d04a1c.2: %Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.1bc: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.d80, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.070f2b.1: %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.070f2b.2: %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.bfb: %LeftShiftAssignWith.type.452 = facet_value %i64, (%LeftShiftAssignWith.impl_witness.1bc) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.c4c: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.bfb) [concrete]
// CHECK:STDOUT:   %.d8e: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.c4c, %LeftShiftAssignWith.facet.bfb [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4240fc.1: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.070f2b.2, @Int.as.LeftShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4240fc.2: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.070f2b.1, @Int.as.LeftShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f17: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.143884.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.904f0d.1: %Int.as.RightShiftAssignWith.impl.Op.type.143884.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.143884.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.904f0d.2: %Int.as.RightShiftAssignWith.impl.Op.type.143884.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.cf7: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.988, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.1daed6.1: %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.1daed6.2: %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.981: %RightShiftAssignWith.type.f17 = facet_value %i64, (%RightShiftAssignWith.impl_witness.cf7) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.c1f: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.981) [concrete]
// CHECK:STDOUT:   %.042: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.c1f, %RightShiftAssignWith.facet.981 [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.19088e.1: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.1daed6.2, @Int.as.RightShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.19088e.2: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.1daed6.1, @Int.as.RightShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.c0b = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a3a = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.c8d [concrete]
// CHECK:STDOUT:   %Core.import_ref.dc62: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.2 (%Int.as.SubAssignWith.impl.Op.type.8ef325.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.2 (constants.%Int.as.SubAssignWith.impl.Op.f183ca.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.ed9 = impl_witness_table (%Core.import_ref.dc62), @Int.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.658: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.1 (%Int.as.SubAssignWith.impl.Op.type.8ef325.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.1 (constants.%Int.as.SubAssignWith.impl.Op.f183ca.2)]
// CHECK:STDOUT:   %Core.import_ref.f8f: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.2 (%Int.as.MulAssignWith.impl.Op.type.99e92a.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.2 (constants.%Int.as.MulAssignWith.impl.Op.1fe857.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.bb2 = impl_witness_table (%Core.import_ref.f8f), @Int.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.241: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.1 (%Int.as.MulAssignWith.impl.Op.type.99e92a.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.1 (constants.%Int.as.MulAssignWith.impl.Op.1fe857.2)]
// CHECK:STDOUT:   %Core.import_ref.8fd: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.2 (%Int.as.DivAssignWith.impl.Op.type.33a652.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.2 (constants.%Int.as.DivAssignWith.impl.Op.e8972a.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.ff8 = impl_witness_table (%Core.import_ref.8fd), @Int.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.0c4: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.1 (%Int.as.DivAssignWith.impl.Op.type.33a652.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.1 (constants.%Int.as.DivAssignWith.impl.Op.e8972a.2)]
// CHECK:STDOUT:   %Core.import_ref.c1a: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.2 (%Int.as.ModAssignWith.impl.Op.type.42e78c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.2 (constants.%Int.as.ModAssignWith.impl.Op.12511c.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.1e5 = impl_witness_table (%Core.import_ref.c1a), @Int.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.820: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.1 (%Int.as.ModAssignWith.impl.Op.type.42e78c.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.1 (constants.%Int.as.ModAssignWith.impl.Op.12511c.2)]
// CHECK:STDOUT:   %Core.import_ref.4a8: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.2 (%Int.as.BitAndAssignWith.impl.Op.type.490e73.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.2 (constants.%Int.as.BitAndAssignWith.impl.Op.c3b0c6.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.b95 = impl_witness_table (%Core.import_ref.4a8), @Int.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.929: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.1 (%Int.as.BitAndAssignWith.impl.Op.type.490e73.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.1 (constants.%Int.as.BitAndAssignWith.impl.Op.c3b0c6.2)]
// CHECK:STDOUT:   %Core.import_ref.5dd: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.2 (%Int.as.BitOrAssignWith.impl.Op.type.f57ae9.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.2 (constants.%Int.as.BitOrAssignWith.impl.Op.d3cc0c.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.80a = impl_witness_table (%Core.import_ref.5dd), @Int.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c4b: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.1 (%Int.as.BitOrAssignWith.impl.Op.type.f57ae9.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.1 (constants.%Int.as.BitOrAssignWith.impl.Op.d3cc0c.2)]
// CHECK:STDOUT:   %Core.import_ref.6fb: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.2 (%Int.as.BitXorAssignWith.impl.Op.type.6fa003.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.2 (constants.%Int.as.BitXorAssignWith.impl.Op.492b4e.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.ed2 = impl_witness_table (%Core.import_ref.6fb), @Int.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f83: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.1 (%Int.as.BitXorAssignWith.impl.Op.type.6fa003.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.1 (constants.%Int.as.BitXorAssignWith.impl.Op.492b4e.2)]
// CHECK:STDOUT:   %Core.import_ref.48b6: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.2 (%Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.2 (constants.%Int.as.LeftShiftAssignWith.impl.Op.d04a1c.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.d80 = impl_witness_table (%Core.import_ref.48b6), @Int.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f1f: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.1 (%Int.as.LeftShiftAssignWith.impl.Op.type.2e97cd.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.1 (constants.%Int.as.LeftShiftAssignWith.impl.Op.d04a1c.2)]
// CHECK:STDOUT:   %Core.import_ref.357: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.2 (%Int.as.RightShiftAssignWith.impl.Op.type.143884.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.2 (constants.%Int.as.RightShiftAssignWith.impl.Op.904f0d.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.988 = impl_witness_table (%Core.import_ref.357), @Int.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.447: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.1 (%Int.as.RightShiftAssignWith.impl.Op.type.143884.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.1 (constants.%Int.as.RightShiftAssignWith.impl.Op.904f0d.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Cpp.long = wrapper_binding b, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ada = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.f16 = impl_witness_access constants.%AddAssignWith.impl_witness.3ce, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.d91 = impl_witness_access constants.%SubAssignWith.impl_witness.9f7, element0 [concrete = constants.%Int.as.SubAssignWith.impl.Op.880561.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Int.as.SubAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.854c3f.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.SubAssignWith.impl.Op.type.a048ce.1 = specific_constant imports.%Core.Op.658, @Int.as.SubAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.SubAssignWith.impl.Op.880561.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Int.as.SubAssignWith.impl.Op.type.a048ce.1 = name_ref Op, %.loc11_5 [concrete = constants.%Int.as.SubAssignWith.impl.Op.880561.1]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Int.as.SubAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.854c3f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Int.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.7b5 = impl_witness_access constants.%MulAssignWith.impl_witness.5a7, element0 [concrete = constants.%Int.as.MulAssignWith.impl.Op.700938.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Int.as.MulAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.e90d66.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.MulAssignWith.impl.Op.type.25fd2b.1 = specific_constant imports.%Core.Op.241, @Int.as.MulAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.MulAssignWith.impl.Op.700938.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Int.as.MulAssignWith.impl.Op.type.25fd2b.1 = name_ref Op, %.loc12_5 [concrete = constants.%Int.as.MulAssignWith.impl.Op.700938.1]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Int.as.MulAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.e90d66.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Int.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %i64 = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.c68 = impl_witness_access constants.%DivAssignWith.impl_witness.39f, element0 [concrete = constants.%Int.as.DivAssignWith.impl.Op.c6930d.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.DivAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.0454de.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.DivAssignWith.impl.Op.type.92ce4f.1 = specific_constant imports.%Core.Op.0c4, @Int.as.DivAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.DivAssignWith.impl.Op.c6930d.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Int.as.DivAssignWith.impl.Op.type.92ce4f.1 = name_ref Op, %.loc13_5 [concrete = constants.%Int.as.DivAssignWith.impl.Op.c6930d.1]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Int.as.DivAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.0454de.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Int.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %i64 = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.360 = impl_witness_access constants.%ModAssignWith.impl_witness.a1c, element0 [concrete = constants.%Int.as.ModAssignWith.impl.Op.b03bdb.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Int.as.ModAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.143374.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.ModAssignWith.impl.Op.type.9b6f30.1 = specific_constant imports.%Core.Op.820, @Int.as.ModAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.ModAssignWith.impl.Op.b03bdb.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Int.as.ModAssignWith.impl.Op.type.9b6f30.1 = name_ref Op, %.loc14_5 [concrete = constants.%Int.as.ModAssignWith.impl.Op.b03bdb.1]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Int.as.ModAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.143374.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Int.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.706 = impl_witness_access constants.%BitAndAssignWith.impl_witness.2c7, element0 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.096e1a.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Int.as.BitAndAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.4f006b.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5: %Int.as.BitAndAssignWith.impl.Op.type.bbaece.1 = specific_constant imports.%Core.Op.929, @Int.as.BitAndAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.096e1a.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Int.as.BitAndAssignWith.impl.Op.type.bbaece.1 = name_ref Op, %.loc16_5 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.096e1a.1]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Int.as.BitAndAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.4f006b.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Int.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16: init %i64 = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %i64 = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.aa5 = impl_witness_access constants.%BitOrAssignWith.impl_witness.084, element0 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.508e1b.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Int.as.BitOrAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.67f0b3.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5: %Int.as.BitOrAssignWith.impl.Op.type.9c3899.1 = specific_constant imports.%Core.Op.c4b, @Int.as.BitOrAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.508e1b.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Int.as.BitOrAssignWith.impl.Op.type.9c3899.1 = name_ref Op, %.loc17_5 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.508e1b.1]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Int.as.BitOrAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.67f0b3.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Int.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17: init %i64 = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %i64 = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.6a6 = impl_witness_access constants.%BitXorAssignWith.impl_witness.8bd, element0 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.e58813.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Int.as.BitXorAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.462aaf.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5: %Int.as.BitXorAssignWith.impl.Op.type.1a4108.1 = specific_constant imports.%Core.Op.f83, @Int.as.BitXorAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.e58813.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Int.as.BitXorAssignWith.impl.Op.type.1a4108.1 = name_ref Op, %.loc18_5 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.e58813.1]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Int.as.BitXorAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.462aaf.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Int.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18: init %i64 = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %i64 = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.d8e = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.1bc, element0 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.070f2b.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Int.as.LeftShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4240fc.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5: %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.1 = specific_constant imports.%Core.Op.f1f, @Int.as.LeftShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.070f2b.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Int.as.LeftShiftAssignWith.impl.Op.type.a787a7.1 = name_ref Op, %.loc19_5 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.070f2b.1]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Int.as.LeftShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4240fc.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Int.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19: init %i64 = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %i64 = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.042 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.cf7, element0 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.1daed6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Int.as.RightShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.19088e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5: %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.1 = specific_constant imports.%Core.Op.447, @Int.as.RightShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.1daed6.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Int.as.RightShiftAssignWith.impl.Op.type.6e18c1.1 = name_ref Op, %.loc20_5 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.1daed6.1]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Int.as.RightShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.19088e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Int.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc20: init %i64 = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc20
// CHECK:STDOUT:   %.loc20_9.2: %i64 = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.5ab: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.1ae: %ImplicitAs.type.340 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.fe0: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.338, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.1: %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.2: %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d04: %AddAssignWith.type.5ab = facet_value %Cpp.long, (%AddAssignWith.impl_witness.fe0) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.90a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.d04) [concrete]
// CHECK:STDOUT:   %.753: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.90a, %AddAssignWith.facet.d04 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f4c4ac.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f4c4ac.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.d3c: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.338 = impl_witness_table (%Core.import_ref.d3c), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.373: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.753 = impl_witness_access constants.%AddAssignWith.impl_witness.fe0, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.5a7) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f4c4ac.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.1 = specific_constant imports.%Core.Op.373, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.6a6c21.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c9f5c8.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f4c4ac.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.3(%a.ref, %.loc9_8.2)
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
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_runtime_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.f57 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.f57 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.091: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6e8: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.091) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.6e8) [concrete]
// CHECK:STDOUT:   %.f34: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a04, %ImplicitAs.facet.6e8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d98: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.210: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.210) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet.210 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e06: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a3a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c71: %ImplicitAs.type.221 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3ce: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.c0b, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.d4b879.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.4484b8.1: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.d4b879.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.4484b8.2: %Int.as.AddAssignWith.impl.Op.type.d4b879.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.2ea: %AddAssignWith.type.e06 = facet_value %i64, (%AddAssignWith.impl_witness.3ce) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.987: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.f16: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.987, %AddAssignWith.facet.2ea [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.4484b8.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.4484b8.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.294: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c71) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.294, %ImplicitAs.facet.c71 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.c0b = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a3a = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.c8d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %i64 = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f34 = impl_witness_access constants.%ImplicitAs.impl_witness.091, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i64 = call %bound_method.loc8_3.2(%int_1.loc8) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8: init %i64 = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %a: ref %i64 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f57 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f57 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Cpp.long = wrapper_binding b, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ada = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.f16 = impl_witness_access constants.%AddAssignWith.impl_witness.3ce, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.d4b879.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.4484b8.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.c71) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b74b59.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9e1 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
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
// CHECK:STDOUT: --- increment_decrement_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.562: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6dd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.562, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.c9e: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.4d7: type = fn_type_with_self_type %Inc.WithSelf.Op.type.c9e, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.6d1: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.974: type = fn_type_with_self_type %Dec.WithSelf.Op.type.6d1, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f73 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.6dd = impl_witness_access constants.%ImplicitAs.impl_witness.5a7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.4d7 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.974 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
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
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Cpp.long) {
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
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long.patt: %pattern_type.b35 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.42f, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.633: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d29: %ImplicitAs.type.14f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9c4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8cc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.d29) [concrete]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8cc, %ImplicitAs.facet.d29 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.633 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.633, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.7e7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.01f: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.130: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7da: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.747: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.54e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dbc: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.747) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.57a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.dbc) [concrete]
// CHECK:STDOUT:   %.869: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.57a, %ImplicitAs.facet.dbc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97f: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_result.patt: %pattern_type.97f = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   %ptr.e40: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %carbon_u64.patt: %pattern_type.b35 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   %carbon_u64_result.patt: %pattern_type.97e = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.68f: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.a83: <witness> = impl_witness imports.%As.impl_witness_table.fcb [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.68f = facet_value Core.IntLiteral, (%As.impl_witness.a83) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d43: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %As.WithSelf.Convert.type.d43, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f9e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d48 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.020: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.f9e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a35: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.020) [concrete]
// CHECK:STDOUT:   %.1d2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a35, %ImplicitAs.facet.020 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.2
// CHECK:STDOUT:     .ULongLongResult = %ULongLongResult.decl
// CHECK:STDOUT:     .PassULongLong = %PassULongLong.cpp_overload_set.value
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     .ULongResult = %ULongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.aa5: @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.170)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.42f = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.54e = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.fcb = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.60e [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d48 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long.as.ImplicitAs.impl.8de [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.93e = impl_witness_access constants.%ImplicitAs.impl_witness.9c4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc15_56.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_56.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %u64 = call %bound_method.loc15_56.2(%int_1.loc15) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc15_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc15_56.2: %u64 = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = wrapper_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.b35 = value_binding_pattern cpp_unsigned_long_long [concrete = constants.%cpp_unsigned_long_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.130 = addr_of %.loc16_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%cpp_unsigned_long_long.ref, %addr.loc16)
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
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc18: %.869 = impl_witness_access constants.%ImplicitAs.impl_witness.747, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.unsigned_long = call %bound_method.loc18(%int_1.loc18) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc18_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc18_46.2: %Cpp.unsigned_long = converted %int_1.loc18, %.loc18_46.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc18_29: type = splice_block %unsigned_long.ref.loc18 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc18: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = wrapper_binding cpp_unsigned_long, %.loc18_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete = constants.%cpp_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc19_93.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.e40 = addr_of %.loc19_93.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.2(%cpp_unsigned_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_93.2: init %ULongResult to %.loc19_93.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_93.3: ref %ULongResult = temporary %.loc19_93.1, %.loc19_93.2
// CHECK:STDOUT:   %.loc19_93.4: %ULongResult = acquire_value %.loc19_93.3
// CHECK:STDOUT:   %.loc19_43: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = wrapper_binding cpp_unsigned_long_result, %.loc19_93.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.97f = value_binding_pattern cpp_unsigned_long_result [concrete = constants.%cpp_unsigned_long_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc21: %.93e = impl_witness_access constants.%ImplicitAs.impl_witness.9c4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %u64.loc21: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_u64: %u64 = wrapper_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.b35 = value_binding_pattern carbon_u64 [concrete = constants.%carbon_u64.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_83.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.130 = addr_of %.loc22_83.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_83.2: init %ULongLongResult to %.loc22_83.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_83.3: ref %ULongLongResult = temporary %.loc22_83.1, %.loc22_83.2
// CHECK:STDOUT:   %.loc22_83.4: %ULongLongResult = acquire_value %.loc22_83.3
// CHECK:STDOUT:   %.loc22_36: type = splice_block %ULongLongResult.ref.loc22 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc22: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_u64_result: %ULongLongResult = wrapper_binding carbon_u64_result, %.loc22_83.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.97e = value_binding_pattern carbon_u64_result [concrete = constants.%carbon_u64_result.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.3e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.c73 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.943]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.c73 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.c73 to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_59.5: init %array_type to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.0ea = impl_witness_access constants.%As.impl_witness.a83, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.1d2 = impl_witness_access constants.%ImplicitAs.impl_witness.f9e, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_83.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc22: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc22(%.loc22_83.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound: <bound method> = bound_method %.loc19_93.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound(%.loc19_93.3)
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
// CHECK:STDOUT: --- copy_unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.172 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.172 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7da: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.747: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.54e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.747) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.57a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.869: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.57a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.172 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.172 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.650: <witness> = impl_witness imports.%Copy.impl_witness_table.ab1 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.650) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6a8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.647: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6a8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.54e = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ab1 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.869 = impl_witness_access constants.%ImplicitAs.impl_witness.747, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.172 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.172 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_37: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.647 = impl_witness_access constants.%Copy.impl_witness.650, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_37, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc9(%.loc9_37)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc9: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.172 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.172 = var_pattern %b.patt [concrete = constants.%b.var_patt]
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
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.unsigned_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
