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

// --- comparisons_heterogeneous_long_and_i32.carbon

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

// --- arithmetic_heterogeneous_long_and_i32.carbon

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

// --- compound_assignment_heterogeneous_long_and_i32.carbon

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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %cpp_long.patt: %pattern_type.ada = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %carbon_long.patt: %pattern_type.277 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c56: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cc8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.85c: %ImplicitAs.type.77d = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e91, %ImplicitAs.facet.85c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.789: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.c37: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.fff: <witness> = impl_witness imports.%As.impl_witness_table.bee [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.c37 = facet_value Core.IntLiteral, (%As.impl_witness.fff) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f09: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.acd: type = fn_type_with_self_type %As.WithSelf.Convert.type.f09, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.250: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a7f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dab: %ImplicitAs.type.7cb = facet_value %Cpp.long, (%ImplicitAs.impl_witness.250) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6c0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.dab) [concrete]
// CHECK:STDOUT:   %.a67: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6c0, %ImplicitAs.facet.dab [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.33f, %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %Array.4a1: type = class_type @Array, @Array(%f32.789, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %Array.4a1 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.66c: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.789)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.610, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e51: %ImplicitAs.type.66c = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.9a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.969: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.789, %ImplicitAs.facet.e51) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.969, %ImplicitAs.facet.e51 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a [concrete]
// CHECK:STDOUT:   %pattern_type.0d4: type = pattern_type %f32.789 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.9ba: %f32.789 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.4a1 = tuple_value (%float.9ba) [concrete]
// CHECK:STDOUT:   %self.param_patt.b45: %pattern_type.0d4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26b: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt.b45 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.115: %pattern_type.951 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b9: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt.115 [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cc8 = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.bee = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.ed4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a7f = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.long.as.ImplicitAs.impl.b7c [concrete]
// CHECK:STDOUT:   %Core.import_ref.f42: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.610 = impl_witness_table (%Core.import_ref.f42), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc11_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc11_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9(%cpp_long.ref)
// CHECK:STDOUT:   %.loc9_33.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_33.2: %i64 = converted %cpp_long.ref, %.loc9_33.1
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = wrapper_binding carbon_long, %.loc9_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.277 = value_binding_pattern carbon_long [concrete = constants.%carbon_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.4a1 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.dd3 = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.789 = call %bound_method.loc11_50.2(%float) [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc11_50.2: init %f32.789 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %f32.789 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %f32.789 to %.loc11_50.3 = in_place_init %.loc11_50.2 [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc11_50.5: init %Array.4a1 to %a.var = array_init (%.loc11_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %Array.4a1 = converted %.loc11_50.1, %.loc11_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_41: type = splice_block %Array [concrete = constants.%Array.4a1] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.789 [concrete = constants.%f32.789]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.acd = impl_witness_access constants.%As.impl_witness.fff, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.a67 = impl_witness_access constants.%ImplicitAs.impl_witness.250, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.789, constants.%int_1.5b8) [concrete = constants.%Array.4a1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.4a1 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.1(%self.param: ref %f32.789) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11: ref %f32.789 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.2(%self.param: ref %Array.4a1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.2(%self.param: ref %Array.4a1) {
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
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7da: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b7: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.75d: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.92f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.75d) [concrete]
// CHECK:STDOUT:   %.39a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.92f, %ImplicitAs.facet.75d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.96f: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_long.patt: %pattern_type.96f = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.755: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9ee [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c56: %ImplicitAs.type.6b7 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.755) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9fd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.c56) [concrete]
// CHECK:STDOUT:   %.b6b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9fd, %ImplicitAs.facet.c56 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.789: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.68f: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.940: <witness> = impl_witness imports.%As.impl_witness_table.d4e [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.68f = facet_value Core.IntLiteral, (%As.impl_witness.940) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ef7: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.20a: type = fn_type_with_self_type %As.WithSelf.Convert.type.ef7, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.34b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.090 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.263: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.34b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.eb5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.263) [concrete]
// CHECK:STDOUT:   %.7d5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.eb5, %ImplicitAs.facet.263 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %Array.4a1: type = class_type @Array, @Array(%f32.789, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %Array.4a1 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.66c: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.789)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.610, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e51: %ImplicitAs.type.66c = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.9a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.969: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.789, %ImplicitAs.facet.e51) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.969, %ImplicitAs.facet.e51 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a [concrete]
// CHECK:STDOUT:   %pattern_type.0d4: type = pattern_type %f32.789 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.9ba: %f32.789 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.4a1 = tuple_value (%float.9ba) [concrete]
// CHECK:STDOUT:   %self.param_patt.b45: %pattern_type.0d4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26b: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt.b45 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.115: %pattern_type.951 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b9: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt.115 [concrete]
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
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9ee = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.unsigned_long.as.ImplicitAs.impl.b6e [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.d4e = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.60e [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.090 = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.unsigned_long.as.ImplicitAs.impl.8de [concrete]
// CHECK:STDOUT:   %Core.import_ref.f42: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.610 = impl_witness_table (%Core.import_ref.f42), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc11_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc11_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.39a = impl_witness_access constants.%ImplicitAs.impl_witness.a75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.b6b = impl_witness_access constants.%ImplicitAs.impl_witness.755, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_unsigned_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9: init %u64 = call %bound_method.loc9(%cpp_unsigned_long.ref)
// CHECK:STDOUT:   %.loc9_42.1: %u64 = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_42.2: %u64 = converted %cpp_unsigned_long.ref, %.loc9_42.1
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = wrapper_binding carbon_unsigned_long, %.loc9_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.96f = value_binding_pattern carbon_unsigned_long [concrete = constants.%carbon_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.4a1 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.dd3 = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.789 = call %bound_method.loc11_59.2(%float) [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc11_59.2: init %f32.789 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_59.3: ref %f32.789 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_59.4: init %f32.789 to %.loc11_59.3 = in_place_init %.loc11_59.2 [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc11_59.5: init %Array.4a1 to %a.var = array_init (%.loc11_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %Array.4a1 = converted %.loc11_59.1, %.loc11_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_50: type = splice_block %Array [concrete = constants.%Array.4a1] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.789 [concrete = constants.%f32.789]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.20a = impl_witness_access constants.%As.impl_witness.940, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.unsigned_long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.7d5 = impl_witness_access constants.%ImplicitAs.impl_witness.34b, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.789, constants.%int_1.5b8) [concrete = constants.%Array.4a1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.4a1 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.1(%self.param: ref %f32.789) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11: ref %f32.789 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.2(%self.param: ref %Array.4a1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.2(%self.param: ref %Array.4a1) {
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %cpp_long_long.patt: %pattern_type.277 = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.353(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.0ab: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.fab: %Int.as.ImplicitAs.impl.Convert.type.0ab = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.248: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.38c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.386: %ImplicitAs.type.77d = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.248) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e1f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.386) [concrete]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e1f, %ImplicitAs.facet.386 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a0e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.dfd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.ccc: %i64 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
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
// CHECK:STDOUT:   %carbon_i64.patt: %pattern_type.277 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   %carbon_i64_result.patt: %pattern_type.357 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   %f32.789: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.9d0: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3d0: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ee3: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.93b: %Core.IntLiteral.as.As.impl.Convert.type.ee3 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.9d0 = facet_value Core.IntLiteral, (%As.impl_witness.3d0) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.c08: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.b3c2: type = fn_type_with_self_type %As.WithSelf.Convert.type.c08, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.93b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.93b, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fe0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.401: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9ad, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.5ff: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.97e: %Int.as.ImplicitAs.impl.Convert.type.5ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b71: %ImplicitAs.type.7cb = facet_value %i64, (%ImplicitAs.impl_witness.401) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.45b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.b71) [concrete]
// CHECK:STDOUT:   %.462: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.45b, %ImplicitAs.facet.b71 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.ccc, %Int.as.ImplicitAs.impl.Convert.97e [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.97e, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.8ae: <bound method> = bound_method %int_1.ccc, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.4a1: type = class_type @Array, @Array(%f32.789, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %Array.4a1 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.66c: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.789)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.610, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e51: %ImplicitAs.type.66c = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.9a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.969: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.789, %ImplicitAs.facet.e51) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.969, %ImplicitAs.facet.e51 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a [concrete]
// CHECK:STDOUT:   %pattern_type.0d4: type = pattern_type %f32.789 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.255: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.9ba: %f32.789 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.4a1 = tuple_value (%float.9ba) [concrete]
// CHECK:STDOUT:   %self.param_patt.b45: %pattern_type.0d4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26b: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt.b45 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.115: %pattern_type.951 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b9: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt.115 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.98a: %pattern_type.357 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.784: %pattern_type.357 = wrapper_binding_pattern self, %self.param_patt.98a [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b62: %pattern_type.ae7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5c45: %pattern_type.ae7 = wrapper_binding_pattern self, %self.param_patt.b62 [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.import_ref.03c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.0ab) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.fab)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9ad = impl_witness_table (%Core.import_ref.03c), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f42: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.610 = impl_witness_table (%Core.import_ref.f42), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.decl: %LongLongResult.cpp_destructor.type = fn_decl @LongLongResult.cpp_destructor [concrete = constants.%LongLongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LongResult.cpp_destructor.decl: %LongResult.cpp_destructor.type = fn_decl @LongResult.cpp_destructor [concrete = constants.%LongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc24_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.357 = ref_param_pattern [concrete = constants.%self.param_patt.98a]
// CHECK:STDOUT:     %self.patt: %pattern_type.357 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.784]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %LongLongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %LongLongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc19 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ae7 = ref_param_pattern [concrete = constants.%self.param_patt.b62]
// CHECK:STDOUT:     %self.patt: %pattern_type.ae7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5c45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %LongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %LongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.71b = impl_witness_access constants.%ImplicitAs.impl_witness.248, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.15f]
// CHECK:STDOUT:   %bound_method.loc15_38.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a0e]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_38.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.dfd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_38.2(%int_1.loc15) [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:   %.loc15_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:   %.loc15_38.2: %i64 = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = wrapper_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.277 = value_binding_pattern cpp_long_long [concrete = constants.%cpp_long_long.patt]
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
// CHECK:STDOUT:   %impl.elem0.loc18: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.277 = value_binding_pattern carbon_i64 [concrete = constants.%carbon_i64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.4a1 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.dd3 = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24_55: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn.loc24_55 [concrete = constants.%bound_method.255]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.789 = call %bound_method.loc24_55.2(%float) [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc24_55.2: init %f32.789 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_55.3: ref %f32.789 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_55.4: init %f32.789 to %.loc24_55.3 = in_place_init %.loc24_55.2 [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc24_55.5: init %Array.4a1 to %a.var = array_init (%.loc24_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %Array.4a1 = converted %.loc24_55.1, %.loc24_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_46: type = splice_block %Array [concrete = constants.%Array.4a1] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.789 [concrete = constants.%f32.789]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.b3c2 = impl_witness_access constants.%As.impl_witness.3d0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.93b]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.1: <specific function> = specific_function %impl.elem0.loc24_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %int_1.loc24, %specific_fn.loc24_30.1 [concrete = constants.%bound_method.fe0]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc24_30.2(%int_1.loc24) [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:     %.loc24_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:     %.loc24_30.2: %i64 = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.462 = impl_witness_access constants.%ImplicitAs.impl_witness.401, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.97e]
// CHECK:STDOUT:     %bound_method.loc24_30.3: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.2: <specific function> = specific_function %impl.elem0.loc24_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.4: <bound method> = bound_method %.loc24_30.2, %specific_fn.loc24_30.2 [concrete = constants.%bound_method.8ae]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.4(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.789, constants.%int_1.5b8) [concrete = constants.%Array.4a1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.4a1 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%a.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_81.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_81.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_73.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_73.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_87.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16(%.loc16_87.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.1(%self.param: ref %f32.789) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24: ref %f32.789 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.2(%self.param: ref %Array.4a1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22(%self.param: ref %LongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22(%self.param: ref %LongLongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%LongLongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc19(%self.param: ref %LongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc19(%self.param: ref %LongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%LongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.78e: <witness> = impl_witness imports.%Copy.impl_witness_table.842 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.78e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.881: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.6f73: type = fn_type_with_self_type %Copy.WithSelf.Op.type.881, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.277 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bc1: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.842 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.6f73 = impl_witness_access constants.%Copy.impl_witness.78e, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %EqWith.type.3ed: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.6bf: <witness> = impl_witness imports.%EqWith.impl_witness_table.e52 [concrete]
// CHECK:STDOUT:   %EqWith.facet.7ae: %EqWith.type.3ed = facet_value %Cpp.long, (%EqWith.impl_witness.6bf) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.5c9: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.7ae) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.8a6: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.7ae) [concrete]
// CHECK:STDOUT:   %.0fad: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.5c9, %EqWith.facet.7ae [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1ca: %Cpp.long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.308: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.8a6, %EqWith.facet.7ae [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a6a: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.484: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.494: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.ce3 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.560: %OrderedWith.type.484 = facet_value %Cpp.long, (%OrderedWith.impl_witness.494) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0f9: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.560) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.2ff: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.560) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d5c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.560) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9ae: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.560) [concrete]
// CHECK:STDOUT:   %.347: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d5c, %OrderedWith.facet.560 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.7c8: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.35c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.0f9, %OrderedWith.facet.560 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.606: %Cpp.long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.aac: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9ae, %OrderedWith.facet.560 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.e3c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.2ff, %OrderedWith.facet.560 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.a87: %Cpp.long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc9: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.e52 = impl_witness_table (%Core.import_ref.a87, %Core.import_ref.adc9), @Cpp.long.as.EqWith.impl.f72 [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e8: %Cpp.long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757ec: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.ce3 = impl_witness_table (%Core.import_ref.0e8, %Core.import_ref.757ec, %Core.import_ref.c32, %Core.import_ref.381), @Cpp.long.as.OrderedWith.impl.f51 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.0fad = impl_witness_access constants.%EqWith.impl_witness.6bf, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.308 = impl_witness_access constants.%EqWith.impl_witness.6bf, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.347 = impl_witness_access constants.%OrderedWith.impl_witness.494, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.35c = impl_witness_access constants.%OrderedWith.impl_witness.494, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.aac = impl_witness_access constants.%OrderedWith.impl_witness.494, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.e3c = impl_witness_access constants.%OrderedWith.impl_witness.494, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.610: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.1ae: %ImplicitAs.type.340 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.477445.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.367ca8.1: %Cpp.long.as.EqWith.impl.NotEqual.type.477445.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee5c37.1: %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.477445.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.367ca8.2: %Cpp.long.as.EqWith.impl.NotEqual.type.477445.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2d(%T.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee5c37.2: %Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.35985e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %T.500: %ImplicitAs.type.35985e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.9e7(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.f66088.1: %T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.69c2e6.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.9e7(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f54c32.1: %T.as_type.as.EqWith.impl.Equal.type.69c2e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.9e7(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.f66088.2: %T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.69c2e6.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.9e7(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f54c32.2: %T.as_type.as.EqWith.impl.Equal.type.69c2e6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.aee: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c56: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cc8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.85c: %ImplicitAs.type.77d = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c56) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.128: <witness> = impl_witness imports.%EqWith.impl_witness_table.d8b, @T.as_type.as.EqWith.impl.9e7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.cc2243.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.9e7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.585450.1: %T.as_type.as.EqWith.impl.Equal.type.cc2243.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.727c77.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.9e7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.384449.1: %T.as_type.as.EqWith.impl.NotEqual.type.727c77.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.cc2243.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.9e7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.585450.2: %T.as_type.as.EqWith.impl.Equal.type.cc2243.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.727c77.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.9e7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.384449.2: %T.as_type.as.EqWith.impl.NotEqual.type.727c77.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.ab4: %EqWith.type.610 = facet_value %Cpp.long, (%EqWith.impl_witness.128) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.7eb: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.ab4) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.83e: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.ab4) [concrete]
// CHECK:STDOUT:   %.360: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.7eb, %EqWith.facet.ab4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.cf1eee.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.585450.2, @T.as_type.as.EqWith.impl.Equal.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.cf1eee.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.585450.1, @T.as_type.as.EqWith.impl.Equal.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e91, %ImplicitAs.facet.85c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.26c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.83e, %EqWith.facet.ab4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.748602.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.384449.2, @T.as_type.as.EqWith.impl.NotEqual.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.748602.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.384449.1, @T.as_type.as.EqWith.impl.NotEqual.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a34: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
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
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.33d696.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a31cf7.1: %T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0a84ea.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.de256e.1: %T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.33d696.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a31cf7.2: %T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0a84ea.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.898(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.de256e.2: %T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.975: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.b07: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.160, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.581e24.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.310b59.1: %T.as_type.as.OrderedWith.impl.Less.type.581e24.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.02a1d7.1: %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.581e24.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.310b59.2: %T.as_type.as.OrderedWith.impl.Less.type.581e24.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.02a1d7.2: %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.898(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.83b: %OrderedWith.type.a34 = facet_value %Cpp.long, (%OrderedWith.impl_witness.b07) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.2d7: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.83b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8fa: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.83b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.322: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.83b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.1b2: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.83b) [concrete]
// CHECK:STDOUT:   %.b20: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.322, %OrderedWith.facet.83b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.631c10.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.02a1d7.2, @T.as_type.as.OrderedWith.impl.Greater.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.631c10.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.02a1d7.1, @T.as_type.as.OrderedWith.impl.Greater.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.5ce: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.2d7, %OrderedWith.facet.83b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.633488.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.310b59.2, @T.as_type.as.OrderedWith.impl.Less.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.633488.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.310b59.1, @T.as_type.as.OrderedWith.impl.Less.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.700: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.1b2, %OrderedWith.facet.83b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8cc398.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8cc398.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.947: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8fa, %OrderedWith.facet.83b [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2741ef.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2741ef.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.325: <witness> = impl_witness imports.%EqWith.impl_witness_table.999, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.b2352a.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.781868.1: %Cpp.long.as.EqWith.impl.Equal.type.b2352a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.044138.1: %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.b2352a.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.781868.2: %Cpp.long.as.EqWith.impl.Equal.type.b2352a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2d(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.044138.2: %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.613: %EqWith.type.aee = facet_value %Cpp.long, (%EqWith.impl_witness.325) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b35: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.613) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.855: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.613) [concrete]
// CHECK:STDOUT:   %.77e: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b35, %EqWith.facet.613 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.49298f.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.781868.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.49298f.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.781868.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.c25: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.855, %EqWith.facet.613 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5594f5.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.044138.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5594f5.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.044138.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f8a: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.8db, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.18f643.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.95dc3a.1: %Cpp.long.as.OrderedWith.impl.Less.type.18f643.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a84228.1: %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.18f643.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.95dc3a.2: %Cpp.long.as.OrderedWith.impl.Less.type.18f643.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a84228.2: %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.d2f(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.1e8: %OrderedWith.type.975 = facet_value %Cpp.long, (%OrderedWith.impl_witness.f8a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.6d3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1e8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.b0c: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1e8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.4c8: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1e8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.e73: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1e8) [concrete]
// CHECK:STDOUT:   %.688: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.4c8, %OrderedWith.facet.1e8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.3b735e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.a84228.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.3b735e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.a84228.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.f54: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.6d3, %OrderedWith.facet.1e8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.3eadcf.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.95dc3a.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.3eadcf.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.95dc3a.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.a87: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.e73, %OrderedWith.facet.1e8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0cf43e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0cf43e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.903: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.b0c, %OrderedWith.facet.1e8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2ca634.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2ca634.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.1af: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.6a18eb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.ee5c37.1)]
// CHECK:STDOUT:   %Core.import_ref.5b6: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.477445.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.367ca8.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.999 = impl_witness_table (%Core.import_ref.1af, %Core.import_ref.5b6), @Cpp.long.as.EqWith.impl.e2d [concrete]
// CHECK:STDOUT:   %Core.Equal.94f: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.6a18eb.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.ee5c37.2)]
// CHECK:STDOUT:   %Core.NotEqual.3eb: @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.477445.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2d.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.367ca8.2)]
// CHECK:STDOUT:   %Core.import_ref.42c: @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.69c2e6.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.f54c32.1)]
// CHECK:STDOUT:   %Core.import_ref.d79: @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.f66088.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.d8b = impl_witness_table (%Core.import_ref.42c, %Core.import_ref.d79), @T.as_type.as.EqWith.impl.9e7 [concrete]
// CHECK:STDOUT:   %Core.Equal.689: @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.69c2e6.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.f54c32.2)]
// CHECK:STDOUT:   %Core.NotEqual.e65: @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.fb7e73.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.9e7.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.f66088.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cc8 = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.49e: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.431fd2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.458c1e.1)]
// CHECK:STDOUT:   %Core.import_ref.5ef: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.1)]
// CHECK:STDOUT:   %Core.import_ref.345: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.029602.1)]
// CHECK:STDOUT:   %Core.import_ref.ccd: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.8db = impl_witness_table (%Core.import_ref.49e, %Core.import_ref.5ef, %Core.import_ref.345, %Core.import_ref.ccd), @Cpp.long.as.OrderedWith.impl.d2f [concrete]
// CHECK:STDOUT:   %Core.Less.890: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.431fd2.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.458c1e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.a1b: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.085d31.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0f4c92.2)]
// CHECK:STDOUT:   %Core.Greater.bd4: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.d7e2b0.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.029602.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.829: @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.4dd52d.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.d2f.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8ca727.2)]
// CHECK:STDOUT:   %Core.import_ref.600: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.de256e.1)]
// CHECK:STDOUT:   %Core.import_ref.30d: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.0a84ea.1)]
// CHECK:STDOUT:   %Core.import_ref.cfc: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.a31cf7.1)]
// CHECK:STDOUT:   %Core.import_ref.cd9: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.33d696.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.160 = impl_witness_table (%Core.import_ref.600, %Core.import_ref.30d, %Core.import_ref.cfc, %Core.import_ref.cd9), @T.as_type.as.OrderedWith.impl.898 [concrete]
// CHECK:STDOUT:   %Core.Less.9a7: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.d4a1ed.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.de256e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.db4: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.cd32e5.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.0a84ea.2)]
// CHECK:STDOUT:   %Core.Greater.b8d: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.37fbe3.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.a31cf7.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.e93: @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.babb97.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.898.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.33d696.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.360 = impl_witness_access constants.%EqWith.impl_witness.128, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.585450.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.cf1eee.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.cc2243.1 = specific_constant imports.%Core.Equal.689, @T.as_type.as.EqWith.impl.9e7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.585450.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.cc2243.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.585450.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.cf1eee.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_3(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %i64 = converted %a.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.26c = impl_witness_access constants.%EqWith.impl_witness.128, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.384449.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.748602.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.727c77.1 = specific_constant imports.%Core.NotEqual.e65, @T.as_type.as.EqWith.impl.9e7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.384449.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.727c77.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.384449.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.748602.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_3(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %i64 = converted %a.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.b20 = impl_witness_access constants.%OrderedWith.impl_witness.b07, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.02a1d7.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.631c10.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.1 = specific_constant imports.%Core.Greater.b8d, @T.as_type.as.OrderedWith.impl.898(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.02a1d7.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.85d1a7.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.02a1d7.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.631c10.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_3(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %i64 = converted %a.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.5ce = impl_witness_access constants.%OrderedWith.impl_witness.b07, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.310b59.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.633488.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.581e24.1 = specific_constant imports.%Core.Less.9a7, @T.as_type.as.OrderedWith.impl.898(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.310b59.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.581e24.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.310b59.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.633488.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_3(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %i64 = converted %a.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.700 = impl_witness_access constants.%OrderedWith.impl_witness.b07, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8cc398.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.1 = specific_constant imports.%Core.GreaterOrEquivalent.e93, @T.as_type.as.OrderedWith.impl.898(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.8967ef.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.845923.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8cc398.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_3(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %i64 = converted %a.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.947 = impl_witness_access constants.%OrderedWith.impl_witness.b07, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2741ef.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.1 = specific_constant imports.%Core.LessOrEquivalent.db4, @T.as_type.as.OrderedWith.impl.898(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.0fcef9.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.3385a3.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2741ef.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_3(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %i64 = converted %a.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %b.ref.loc15)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.77e = impl_witness_access constants.%EqWith.impl_witness.325, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.781868.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.49298f.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.EqWith.impl.Equal.type.b2352a.1 = specific_constant imports.%Core.Equal.94f, @Cpp.long.as.EqWith.impl.e2d(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.781868.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.b2352a.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.781868.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.49298f.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.c25 = impl_witness_access constants.%EqWith.impl_witness.325, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.044138.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5594f5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.1 = specific_constant imports.%Core.NotEqual.3eb, @Cpp.long.as.EqWith.impl.e2d(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.044138.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.e86037.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.044138.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5594f5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.688 = impl_witness_access constants.%OrderedWith.impl_witness.f8a, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.a84228.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.3b735e.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.1 = specific_constant imports.%Core.Greater.bd4, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.a84228.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.41c64a.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.a84228.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.3b735e.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.f54 = impl_witness_access constants.%OrderedWith.impl_witness.f8a, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.95dc3a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.3eadcf.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.18f643.1 = specific_constant imports.%Core.Less.890, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.95dc3a.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.18f643.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.95dc3a.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.3eadcf.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.a87 = impl_witness_access constants.%OrderedWith.impl_witness.f8a, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0cf43e.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.1 = specific_constant imports.%Core.GreaterOrEquivalent.829, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.985513.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.82f8f7.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.0cf43e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc21: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.903 = impl_witness_access constants.%OrderedWith.impl_witness.f8a, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2ca634.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.1 = specific_constant imports.%Core.LessOrEquivalent.a1b, @Cpp.long.as.OrderedWith.impl.d2f(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3f6f6d.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1384c0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.b83) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2ca634.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc22: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.35985e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %T.500: %ImplicitAs.type.35985e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.668a93.1: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.e92(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.240c37.1: %Int.as.EqWith.impl.NotEqual.type.668a93.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.b39df5.1: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.e92(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.d42841.1: %Int.as.EqWith.impl.Equal.type.b39df5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.668a93.2: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.e92(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.240c37.2: %Int.as.EqWith.impl.NotEqual.type.668a93.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.b39df5.2: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.e92(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.d42841.2: %Int.as.EqWith.impl.Equal.type.b39df5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c56: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cc8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.85c: %ImplicitAs.type.77d = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c56) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.057: <witness> = impl_witness imports.%EqWith.impl_witness_table.b82, @Int.as.EqWith.impl.e92(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.09c0fe.1: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.e92(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.6e4f20.1: %Int.as.EqWith.impl.Equal.type.09c0fe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.9538ce.1: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.e92(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.4aef08.1: %Int.as.EqWith.impl.NotEqual.type.9538ce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.09c0fe.2: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.e92(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.6e4f20.2: %Int.as.EqWith.impl.Equal.type.09c0fe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.9538ce.2: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.e92(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.4aef08.2: %Int.as.EqWith.impl.NotEqual.type.9538ce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.af6: %EqWith.type.3ed = facet_value %i64, (%EqWith.impl_witness.057) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.8e3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.af6) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.6a4: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.af6) [concrete]
// CHECK:STDOUT:   %.74c4: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.8e3, %EqWith.facet.af6 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.3fca43.1: <specific function> = specific_function %Int.as.EqWith.impl.Equal.6e4f20.2, @Int.as.EqWith.impl.Equal.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.3fca43.2: <specific function> = specific_function %Int.as.EqWith.impl.Equal.6e4f20.1, @Int.as.EqWith.impl.Equal.3(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e91, %ImplicitAs.facet.85c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.69e: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.6a4, %EqWith.facet.af6 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.5865c1.1: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.4aef08.2, @Int.as.EqWith.impl.NotEqual.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.5865c1.2: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.4aef08.1, @Int.as.EqWith.impl.NotEqual.3(%int_64, %ImplicitAs.facet.85c) [concrete]
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
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.fd71ab.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.30e13b.1: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.dea463.1: %Int.as.OrderedWith.impl.Greater.type.30e13b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.a41254.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.fe7eba.1: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bed27e.1: %Int.as.OrderedWith.impl.Less.type.fe7eba.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.fd71ab.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.30e13b.2: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.dea463.2: %Int.as.OrderedWith.impl.Greater.type.30e13b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.a41254.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.fe7eba.2: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.c67(%N, %T.500) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bed27e.2: %Int.as.OrderedWith.impl.Less.type.fe7eba.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.c5b: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.216, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.3d9f99.1: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.689225.1: %Int.as.OrderedWith.impl.Less.type.3d9f99.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.e89abd.1: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.7ba760.1: %Int.as.OrderedWith.impl.Greater.type.e89abd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.3d9f99.2: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.689225.2: %Int.as.OrderedWith.impl.Less.type.3d9f99.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.e89abd.2: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.7ba760.2: %Int.as.OrderedWith.impl.Greater.type.e89abd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.c67(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.717: %OrderedWith.type.484 = facet_value %i64, (%OrderedWith.impl_witness.c5b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ac2: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.717) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.28c: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.717) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.a6b: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.717) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.dd4: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.717) [concrete]
// CHECK:STDOUT:   %.c7b: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.a6b, %OrderedWith.facet.717 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.109c49.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.7ba760.2, @Int.as.OrderedWith.impl.Greater.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.109c49.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.7ba760.1, @Int.as.OrderedWith.impl.Greater.3(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.d03: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ac2, %OrderedWith.facet.717 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.8af9a3.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.689225.2, @Int.as.OrderedWith.impl.Less.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.8af9a3.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.689225.1, @Int.as.OrderedWith.impl.Less.3(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.d14: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.dd4, %OrderedWith.facet.717 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f0ae27.1: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.2, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f0ae27.2: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.1, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.69d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.28c, %OrderedWith.facet.717 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.d5052b.1: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.2, @Int.as.OrderedWith.impl.LessOrEquivalent.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.d5052b.2: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.1, @Int.as.OrderedWith.impl.LessOrEquivalent.3(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.95b: <witness> = impl_witness imports.%EqWith.impl_witness_table.896, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.0a8f82.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.86b2db.1: %T.as_type.as.EqWith.impl.Equal.type.0a8f82.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.49c805.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.ca2900.1: %T.as_type.as.EqWith.impl.NotEqual.type.49c805.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.0a8f82.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.86b2db.2: %T.as_type.as.EqWith.impl.Equal.type.0a8f82.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.49c805.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.7e7(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.ca2900.2: %T.as_type.as.EqWith.impl.NotEqual.type.49c805.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.cf5: %EqWith.type.3ed = facet_value Core.IntLiteral, (%EqWith.impl_witness.95b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.e8f: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.cf5) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.83a: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.cf5) [concrete]
// CHECK:STDOUT:   %.dd8: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.e8f, %EqWith.facet.cf5 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ca7715.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.86b2db.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.634293.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.86b2db.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.e608d8.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.634293.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ca7715.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.86b2db.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.634293.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.86b2db.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.e608d8.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.634293.2 [concrete]
// CHECK:STDOUT:   %.865: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.83a, %EqWith.facet.cf5 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.cfadf3.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.ca2900.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.ca2900.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.989d3b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.cfadf3.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.ca2900.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.ca2900.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.989d3b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.407: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.9a7, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.90bf59.1: %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.923208.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.06beca.1: %T.as_type.as.OrderedWith.impl.Greater.type.923208.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.90bf59.2: %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.923208.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.06beca.2: %T.as_type.as.OrderedWith.impl.Greater.type.923208.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.4ed(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.1c0: %OrderedWith.type.484 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.407) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.04f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.1c0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.258: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.1c0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.294: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.1c0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.de5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.1c0) [concrete]
// CHECK:STDOUT:   %.e84: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.294, %OrderedWith.facet.1c0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.a7b8bc.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.06beca.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.06beca.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.3fa851.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.a7b8bc.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.06beca.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.06beca.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.3fa851.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.2 [concrete]
// CHECK:STDOUT:   %.fd0: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.04f, %OrderedWith.facet.1c0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.32a3f6.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.90bf59.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.90bf59.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.70aaf6.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.32a3f6.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.90bf59.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.90bf59.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.70aaf6.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.2 [concrete]
// CHECK:STDOUT:   %.bf0: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.de5, %OrderedWith.facet.1c0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.6cb83f.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.d6150d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.6cb83f.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.d6150d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.2 [concrete]
// CHECK:STDOUT:   %.4d3: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.258, %OrderedWith.facet.1c0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.f6c834.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.1cd393.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.f6c834.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %bound_method.1cd393.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.ad0: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.f384d8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.99b5a3.1)]
// CHECK:STDOUT:   %Core.import_ref.8c0: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.4349fe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.50a028.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.896 = impl_witness_table (%Core.import_ref.ad0, %Core.import_ref.8c0), @T.as_type.as.EqWith.impl.7e7 [concrete]
// CHECK:STDOUT:   %Core.Equal.c5f: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.f384d8.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.99b5a3.2)]
// CHECK:STDOUT:   %Core.NotEqual.e58: @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.4349fe.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.7e7.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.50a028.2)]
// CHECK:STDOUT:   %Core.import_ref.8d10: @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.Equal.type.2 (%Int.as.EqWith.impl.Equal.type.b39df5.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.Equal.2 (constants.%Int.as.EqWith.impl.Equal.d42841.1)]
// CHECK:STDOUT:   %Core.import_ref.0df: @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.NotEqual.type.2 (%Int.as.EqWith.impl.NotEqual.type.668a93.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.NotEqual.2 (constants.%Int.as.EqWith.impl.NotEqual.240c37.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.b82 = impl_witness_table (%Core.import_ref.8d10, %Core.import_ref.0df), @Int.as.EqWith.impl.e92 [concrete]
// CHECK:STDOUT:   %Core.Equal.a50: @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.Equal.type.1 (%Int.as.EqWith.impl.Equal.type.b39df5.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.Equal.1 (constants.%Int.as.EqWith.impl.Equal.d42841.2)]
// CHECK:STDOUT:   %Core.NotEqual.21d: @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.NotEqual.type.1 (%Int.as.EqWith.impl.NotEqual.type.668a93.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @Int.as.EqWith.impl.e92.%Int.as.EqWith.impl.NotEqual.1 (constants.%Int.as.EqWith.impl.NotEqual.240c37.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cc8 = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.011: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.9ca225.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.571834.1)]
// CHECK:STDOUT:   %Core.import_ref.ba2: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.1)]
// CHECK:STDOUT:   %Core.import_ref.44d: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.776eb1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.bf137f.1)]
// CHECK:STDOUT:   %Core.import_ref.3aa: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.9a7 = impl_witness_table (%Core.import_ref.011, %Core.import_ref.ba2, %Core.import_ref.44d, %Core.import_ref.3aa), @T.as_type.as.OrderedWith.impl.4ed [concrete]
// CHECK:STDOUT:   %Core.Less.1d9: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.9ca225.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.571834.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.88c: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3b16e8.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fd69b3.2)]
// CHECK:STDOUT:   %Core.Greater.172: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.776eb1.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.bf137f.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.09a: @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.985ed6.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.4ed.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.17b48a.2)]
// CHECK:STDOUT:   %Core.import_ref.133: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Less.type.2 (%Int.as.OrderedWith.impl.Less.type.fe7eba.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Less.2 (constants.%Int.as.OrderedWith.impl.Less.bed27e.1)]
// CHECK:STDOUT:   %Core.import_ref.30fa: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.a41254.1)]
// CHECK:STDOUT:   %Core.import_ref.217: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Greater.type.2 (%Int.as.OrderedWith.impl.Greater.type.30e13b.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Greater.2 (constants.%Int.as.OrderedWith.impl.Greater.dea463.1)]
// CHECK:STDOUT:   %Core.import_ref.de83: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.fd71ab.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.216 = impl_witness_table (%Core.import_ref.133, %Core.import_ref.30fa, %Core.import_ref.217, %Core.import_ref.de83), @Int.as.OrderedWith.impl.c67 [concrete]
// CHECK:STDOUT:   %Core.Less.2a1: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Less.type.1 (%Int.as.OrderedWith.impl.Less.type.fe7eba.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Less.1 (constants.%Int.as.OrderedWith.impl.Less.bed27e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.113: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.65ffe7.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.a41254.2)]
// CHECK:STDOUT:   %Core.Greater.8c7: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Greater.type.1 (%Int.as.OrderedWith.impl.Greater.type.30e13b.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.Greater.1 (constants.%Int.as.OrderedWith.impl.Greater.dea463.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.33c: @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.a20aa0.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.c67.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.fd71ab.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.74c4 = impl_witness_access constants.%EqWith.impl_witness.057, element0 [concrete = constants.%Int.as.EqWith.impl.Equal.6e4f20.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.EqWith.impl.Equal.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.3fca43.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.EqWith.impl.Equal.type.09c0fe.1 = specific_constant imports.%Core.Equal.a50, @Int.as.EqWith.impl.e92(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.Equal.6e4f20.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Int.as.EqWith.impl.Equal.type.09c0fe.1 = name_ref Equal, %.loc10_5 [concrete = constants.%Int.as.EqWith.impl.Equal.6e4f20.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @Int.as.EqWith.impl.Equal.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.3fca43.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %Int.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %a.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%b.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.69e = impl_witness_access constants.%EqWith.impl_witness.057, element1 [concrete = constants.%Int.as.EqWith.impl.NotEqual.4aef08.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Int.as.EqWith.impl.NotEqual.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.5865c1.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.EqWith.impl.NotEqual.type.9538ce.1 = specific_constant imports.%Core.NotEqual.21d, @Int.as.EqWith.impl.e92(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.NotEqual.4aef08.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Int.as.EqWith.impl.NotEqual.type.9538ce.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%Int.as.EqWith.impl.NotEqual.4aef08.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @Int.as.EqWith.impl.NotEqual.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.5865c1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %Int.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %a.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%b.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.c7b = impl_witness_access constants.%OrderedWith.impl_witness.c5b, element2 [concrete = constants.%Int.as.OrderedWith.impl.Greater.7ba760.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Int.as.OrderedWith.impl.Greater.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.109c49.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.OrderedWith.impl.Greater.type.e89abd.1 = specific_constant imports.%Core.Greater.8c7, @Int.as.OrderedWith.impl.c67(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Greater.7ba760.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Int.as.OrderedWith.impl.Greater.type.e89abd.1 = name_ref Greater, %.loc12_5 [concrete = constants.%Int.as.OrderedWith.impl.Greater.7ba760.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @Int.as.OrderedWith.impl.Greater.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.109c49.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %Int.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_7(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %i64 = converted %a.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%b.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.d03 = impl_witness_access constants.%OrderedWith.impl_witness.c5b, element0 [concrete = constants.%Int.as.OrderedWith.impl.Less.689225.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.OrderedWith.impl.Less.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.8af9a3.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.OrderedWith.impl.Less.type.3d9f99.1 = specific_constant imports.%Core.Less.2a1, @Int.as.OrderedWith.impl.c67(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Less.689225.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Int.as.OrderedWith.impl.Less.type.3d9f99.1 = name_ref Less, %.loc13_5 [concrete = constants.%Int.as.OrderedWith.impl.Less.689225.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @Int.as.OrderedWith.impl.Less.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.8af9a3.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %Int.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_7(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %i64 = converted %a.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%b.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.d14 = impl_witness_access constants.%OrderedWith.impl_witness.c5b, element3 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f0ae27.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.1 = specific_constant imports.%Core.GreaterOrEquivalent.33c, @Int.as.OrderedWith.impl.c67(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.6fea56.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.b9eabe.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f0ae27.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %a.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%b.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.69d = impl_witness_access constants.%OrderedWith.impl_witness.c5b, element1 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.d5052b.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.1 = specific_constant imports.%Core.LessOrEquivalent.113, @Int.as.OrderedWith.impl.c67(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Int.as.OrderedWith.impl.LessOrEquivalent.type.8e4d2e.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.2c6d0d.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.d5052b.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_8(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %i64 = converted %a.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%b.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.dd8 = impl_witness_access constants.%EqWith.impl_witness.95b, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.86b2db.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ca7715.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.634293.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.e608d8.1]
// CHECK:STDOUT:   %.loc17_5.3: %T.as_type.as.EqWith.impl.Equal.type.0a8f82.1 = specific_constant imports.%Core.Equal.c5f, @T.as_type.as.EqWith.impl.7e7(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.86b2db.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.0a8f82.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.86b2db.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ca7715.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.634293.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn [concrete = constants.%bound_method.e608d8.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.865 = impl_witness_access constants.%EqWith.impl_witness.95b, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.ca2900.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.cfadf3.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.989d3b.1]
// CHECK:STDOUT:   %.loc18_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.49c805.1 = specific_constant imports.%Core.NotEqual.e58, @T.as_type.as.EqWith.impl.7e7(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.ca2900.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.49c805.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.ca2900.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.cfadf3.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.d1a411.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn [concrete = constants.%bound_method.989d3b.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.e84 = impl_witness_access constants.%OrderedWith.impl_witness.407, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.06beca.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.a7b8bc.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.3fa851.1]
// CHECK:STDOUT:   %.loc19_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.923208.1 = specific_constant imports.%Core.Greater.172, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.06beca.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.923208.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.06beca.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.a7b8bc.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.db0422.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn [concrete = constants.%bound_method.3fa851.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.fd0 = impl_witness_access constants.%OrderedWith.impl_witness.407, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.90bf59.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.32a3f6.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.70aaf6.1]
// CHECK:STDOUT:   %.loc20_5.3: %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.1 = specific_constant imports.%Core.Less.1d9, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.90bf59.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.92c3cf.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.90bf59.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.32a3f6.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.da4570.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn [concrete = constants.%bound_method.70aaf6.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.bf0 = impl_witness_access constants.%OrderedWith.impl_witness.407, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.6cb83f.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.d6150d.1]
// CHECK:STDOUT:   %.loc21_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.1 = specific_constant imports.%Core.GreaterOrEquivalent.09a, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7d5e6b.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.265add.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.6cb83f.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.95bfcf.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn [concrete = constants.%bound_method.d6150d.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.4d3 = impl_witness_access constants.%OrderedWith.impl_witness.407, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.f6c834.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.b83]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.1cd393.1]
// CHECK:STDOUT:   %.loc22_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.1 = specific_constant imports.%Core.LessOrEquivalent.88c, @T.as_type.as.OrderedWith.impl.4ed(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e2b20e.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.0ab3f0.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.f6c834.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.b83) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a01051.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn [concrete = constants.%bound_method.1cd393.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.6c4: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.61d: %T.as_type.as.ImplicitAs.impl.Convert.type.6c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
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
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.78e: <witness> = impl_witness imports.%Copy.impl_witness_table.842 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.78e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0e1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fac, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0ac: %ImplicitAs.type.340 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.0e1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.26d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.2c1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1f82a6.1: %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.1f82a6.2: %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.008: %AddAssignWith.type.e06 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.26d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.ddd: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.008) [concrete]
// CHECK:STDOUT:   %.eef: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.ddd, %AddAssignWith.facet.008 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c73085.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.1f82a6.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c73085.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.1f82a6.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.6ba: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c163a1.1: %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c163a1.2: %Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.4a6: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.fec, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fbc431.1: %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fbc431.2: %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.15d: %SubAssignWith.type.6ba = facet_value %Cpp.long, (%SubAssignWith.impl_witness.4a6) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.218: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.15d) [concrete]
// CHECK:STDOUT:   %.655: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.218, %SubAssignWith.facet.15d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.662a88.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.fbc431.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.662a88.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.fbc431.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.642: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5870c9.1: %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5870c9.2: %Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.440: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.6f8, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.1: %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.2: %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.d29: %MulAssignWith.type.642 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.440) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3d3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.d29) [concrete]
// CHECK:STDOUT:   %.bc6: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3d3, %MulAssignWith.facet.d29 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.99fbf8.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.99fbf8.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.64b: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.302d61.1: %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.302d61.2: %Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.388: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.d74, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.1: %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.2: %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.358: %DivAssignWith.type.64b = facet_value %Cpp.long, (%DivAssignWith.impl_witness.388) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.027: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.358) [concrete]
// CHECK:STDOUT:   %.7c8: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.027, %DivAssignWith.facet.358 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ab7990.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ab7990.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.132: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.d0e55c.1: %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.d0e55c.2: %Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.bf2: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.139, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.c6a637.1: %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.c6a637.2: %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.c96: %ModAssignWith.type.132 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.bf2) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.092: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.c96) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.092, %ModAssignWith.facet.c96 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.457f6e.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.c6a637.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.457f6e.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.c6a637.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.44e: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.999389.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.999389.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.66f: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.c17, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.11d: %BitAndAssignWith.type.44e = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.66f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.399: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.11d) [concrete]
// CHECK:STDOUT:   %.696: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.399, %BitAndAssignWith.facet.11d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.76c5c3.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.76c5c3.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.e7d: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.980: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.65b, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.5ef: %BitOrAssignWith.type.e7d = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.980) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.85c: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.5ef) [concrete]
// CHECK:STDOUT:   %.622: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.85c, %BitOrAssignWith.facet.5ef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2cb4f5.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2cb4f5.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ce1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.158: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.360, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.7add11.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.7add11.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.a89: %BitXorAssignWith.type.ce1 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.158) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.896: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.a89) [concrete]
// CHECK:STDOUT:   %.463: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.896, %BitXorAssignWith.facet.a89 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.51e38d.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.7add11.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.51e38d.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.7add11.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.452: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.e1c: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.39a, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.cf7: %LeftShiftAssignWith.type.452 = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.e1c) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.d68: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.cf7) [concrete]
// CHECK:STDOUT:   %.134: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.d68, %LeftShiftAssignWith.facet.cf7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.5e9221.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.5e9221.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f17: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.1ae) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.7f8: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.435, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.af7: %RightShiftAssignWith.type.f17 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.7f8) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.efa: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.af7) [concrete]
// CHECK:STDOUT:   %.934: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.efa, %RightShiftAssignWith.facet.af7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.368c78.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.368c78.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.0ac) [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.277 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bc1: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fac = impl_witness_table (%Core.import_ref.051), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.d3c: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.2c1 = impl_witness_table (%Core.import_ref.d3c), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.373: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.842 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f7887: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c163a1.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.fec = impl_witness_table (%Core.import_ref.f7887), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.789: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.24d48a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c163a1.2)]
// CHECK:STDOUT:   %Core.import_ref.094: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.5870c9.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.6f8 = impl_witness_table (%Core.import_ref.094), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b2c: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.aef5ec.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.5870c9.2)]
// CHECK:STDOUT:   %Core.import_ref.a6a: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.302d61.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.d74 = impl_witness_table (%Core.import_ref.a6a), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f8d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.5f53e2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.302d61.2)]
// CHECK:STDOUT:   %Core.import_ref.56b: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.82500c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.d0e55c.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.139 = impl_witness_table (%Core.import_ref.56b), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6f1: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.82500c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.d0e55c.2)]
// CHECK:STDOUT:   %Core.import_ref.130: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.999389.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.130), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.09f: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a75e25.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.999389.2)]
// CHECK:STDOUT:   %Core.import_ref.839: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.65b = impl_witness_table (%Core.import_ref.839), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.607: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.075426.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.2cf22a.2)]
// CHECK:STDOUT:   %Core.import_ref.39e: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.360 = impl_witness_table (%Core.import_ref.39e), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aed: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.5d04ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d86e90.2)]
// CHECK:STDOUT:   %Core.import_ref.7f9: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.39a = impl_witness_table (%Core.import_ref.7f9), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.114: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.d0d9a1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.74ba28.2)]
// CHECK:STDOUT:   %Core.import_ref.919: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.435 = impl_witness_table (%Core.import_ref.919), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.592: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.dd27f0.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.9a6679.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.eef = impl_witness_access constants.%AddAssignWith.impl_witness.26d, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.1f82a6.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c73085.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.1 = specific_constant imports.%Core.Op.373, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.1f82a6.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.bf13e2.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.1f82a6.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c73085.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.655 = impl_witness_access constants.%SubAssignWith.impl_witness.4a6, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fbc431.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.662a88.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.1 = specific_constant imports.%Core.Op.789, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fbc431.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.1e8580.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fbc431.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.662a88.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.bc6 = impl_witness_access constants.%MulAssignWith.impl_witness.440, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.99fbf8.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.1 = specific_constant imports.%Core.Op.b2c, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.2c2434.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.2e1fc3.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.99fbf8.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.7c8 = impl_witness_access constants.%DivAssignWith.impl_witness.388, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ab7990.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.1 = specific_constant imports.%Core.Op.f8d, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.94e95a.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a6c1ca.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ab7990.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.737 = impl_witness_access constants.%ModAssignWith.impl_witness.bf2, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c6a637.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.457f6e.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.1 = specific_constant imports.%Core.Op.6f1, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c6a637.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.915f0b.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.c6a637.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.457f6e.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.696 = impl_witness_access constants.%BitAndAssignWith.impl_witness.66f, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.76c5c3.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.1 = specific_constant imports.%Core.Op.09f, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.583d50.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2b3ee9.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.76c5c3.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.622 = impl_witness_access constants.%BitOrAssignWith.impl_witness.980, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2cb4f5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.1 = specific_constant imports.%Core.Op.607, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.6c8d67.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.fe9f4e.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2cb4f5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.463 = impl_witness_access constants.%BitXorAssignWith.impl_witness.158, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.7add11.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.51e38d.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.1 = specific_constant imports.%Core.Op.aed, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.7add11.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6882b1.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.7add11.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.51e38d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.134 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.e1c, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.5e9221.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.1 = specific_constant imports.%Core.Op.114, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.56f076.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.d57ba2.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.5e9221.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.934 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.7f8, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.340 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.0e1) [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.340 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.0ac]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.368c78.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.1 = specific_constant imports.%Core.Op.592, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.62007c.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.3dae5a.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0ac) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.368c78.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e06: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.35985e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %U.500: %ImplicitAs.type.35985e.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.1: %Int.as.AddAssignWith.impl.Op.type.e95403.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.2: %Int.as.AddAssignWith.impl.Op.type.e95403.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c56: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cc8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.85c: %ImplicitAs.type.77d = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c56) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.7ac: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.16c, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.35824f.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.1c439d.1: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.35824f.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.1c439d.2: %Int.as.AddAssignWith.impl.Op.type.35824f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.4d3: %AddAssignWith.type.e06 = facet_value %i64, (%AddAssignWith.impl_witness.7ac) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d4a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.4d3) [concrete]
// CHECK:STDOUT:   %.34c: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d4a, %AddAssignWith.facet.4d3 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.1c439d.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.1c439d.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e91, %ImplicitAs.facet.85c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.6ba: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.b78b27.1: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.e5d778.1: %Int.as.SubAssignWith.impl.Op.type.b78b27.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.b78b27.2: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.e5d778.2: %Int.as.SubAssignWith.impl.Op.type.b78b27.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.e52: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.a47, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.16701f.1: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.294900.1: %Int.as.SubAssignWith.impl.Op.type.16701f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.16701f.2: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.294900.2: %Int.as.SubAssignWith.impl.Op.type.16701f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.770: %SubAssignWith.type.6ba = facet_value %i64, (%SubAssignWith.impl_witness.e52) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.ff4: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.770) [concrete]
// CHECK:STDOUT:   %.1e9: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.ff4, %SubAssignWith.facet.770 [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.53c9b3.1: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.294900.2, @Int.as.SubAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.53c9b3.2: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.294900.1, @Int.as.SubAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.642: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.dfe9fc.1: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.64e092.1: %Int.as.MulAssignWith.impl.Op.type.dfe9fc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.dfe9fc.2: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.64e092.2: %Int.as.MulAssignWith.impl.Op.type.dfe9fc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.289: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.ef2, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.2a93dc.1: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.c28854.1: %Int.as.MulAssignWith.impl.Op.type.2a93dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.2a93dc.2: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.c28854.2: %Int.as.MulAssignWith.impl.Op.type.2a93dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.d41: %MulAssignWith.type.642 = facet_value %i64, (%MulAssignWith.impl_witness.289) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3ae: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.d41) [concrete]
// CHECK:STDOUT:   %.d13: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3ae, %MulAssignWith.facet.d41 [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.f89e08.1: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.c28854.2, @Int.as.MulAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.f89e08.2: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.c28854.1, @Int.as.MulAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.64b: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.24ad43.1: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.d4f4ad.1: %Int.as.DivAssignWith.impl.Op.type.24ad43.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.24ad43.2: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.d4f4ad.2: %Int.as.DivAssignWith.impl.Op.type.24ad43.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.503: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.78c, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.cc5ddf.1: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.3415fd.1: %Int.as.DivAssignWith.impl.Op.type.cc5ddf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.cc5ddf.2: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.3415fd.2: %Int.as.DivAssignWith.impl.Op.type.cc5ddf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.498: %DivAssignWith.type.64b = facet_value %i64, (%DivAssignWith.impl_witness.503) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.b38: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.498) [concrete]
// CHECK:STDOUT:   %.9e5: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.b38, %DivAssignWith.facet.498 [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.43b329.1: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.3415fd.2, @Int.as.DivAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.43b329.2: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.3415fd.1, @Int.as.DivAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.132: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.c4c096.1: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.568a64.1: %Int.as.ModAssignWith.impl.Op.type.c4c096.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.c4c096.2: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.568a64.2: %Int.as.ModAssignWith.impl.Op.type.c4c096.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.e02: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.8a0, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.977d70.1: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.293f95.1: %Int.as.ModAssignWith.impl.Op.type.977d70.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.977d70.2: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.293f95.2: %Int.as.ModAssignWith.impl.Op.type.977d70.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.659: %ModAssignWith.type.132 = facet_value %i64, (%ModAssignWith.impl_witness.e02) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.6f2: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.659) [concrete]
// CHECK:STDOUT:   %.300: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.6f2, %ModAssignWith.facet.659 [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.68356e.1: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.293f95.2, @Int.as.ModAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.68356e.2: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.293f95.1, @Int.as.ModAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.44e: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.b67ba2.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.af7879.1: %Int.as.BitAndAssignWith.impl.Op.type.b67ba2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.b67ba2.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.af7879.2: %Int.as.BitAndAssignWith.impl.Op.type.b67ba2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.72d: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.c62, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.026e1b.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.717a47.1: %Int.as.BitAndAssignWith.impl.Op.type.026e1b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.026e1b.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.717a47.2: %Int.as.BitAndAssignWith.impl.Op.type.026e1b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.6db: %BitAndAssignWith.type.44e = facet_value %i64, (%BitAndAssignWith.impl_witness.72d) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.c82: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.6db) [concrete]
// CHECK:STDOUT:   %.615: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.c82, %BitAndAssignWith.facet.6db [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.fb3f3f.1: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.717a47.2, @Int.as.BitAndAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.fb3f3f.2: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.717a47.1, @Int.as.BitAndAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.e7d: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.9f45aa.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.d861a8.1: %Int.as.BitOrAssignWith.impl.Op.type.9f45aa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.9f45aa.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.d861a8.2: %Int.as.BitOrAssignWith.impl.Op.type.9f45aa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.6f2: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.c22, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.1c2315.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.9e9473.1: %Int.as.BitOrAssignWith.impl.Op.type.1c2315.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.1c2315.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.9e9473.2: %Int.as.BitOrAssignWith.impl.Op.type.1c2315.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.b5a: %BitOrAssignWith.type.e7d = facet_value %i64, (%BitOrAssignWith.impl_witness.6f2) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.761: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.b5a) [concrete]
// CHECK:STDOUT:   %.491: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.761, %BitOrAssignWith.facet.b5a [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.741d42.1: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.9e9473.2, @Int.as.BitOrAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.741d42.2: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.9e9473.1, @Int.as.BitOrAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ce1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.cccbae.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.ffde62.1: %Int.as.BitXorAssignWith.impl.Op.type.cccbae.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.cccbae.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.ffde62.2: %Int.as.BitXorAssignWith.impl.Op.type.cccbae.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.11a: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.e49, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.02370f.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.5b03e1.1: %Int.as.BitXorAssignWith.impl.Op.type.02370f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.02370f.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.5b03e1.2: %Int.as.BitXorAssignWith.impl.Op.type.02370f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.7e3: %BitXorAssignWith.type.ce1 = facet_value %i64, (%BitXorAssignWith.impl_witness.11a) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.05e: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.7e3) [concrete]
// CHECK:STDOUT:   %.3f5: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.05e, %BitXorAssignWith.facet.7e3 [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.b395e5.1: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.5b03e1.2, @Int.as.BitXorAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.b395e5.2: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.5b03e1.1, @Int.as.BitXorAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.452: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.42b491.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.dadf31.1: %Int.as.LeftShiftAssignWith.impl.Op.type.42b491.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.42b491.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.dadf31.2: %Int.as.LeftShiftAssignWith.impl.Op.type.42b491.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.710: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.d49, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.33ee4d.1: %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.33ee4d.2: %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.f8b: %LeftShiftAssignWith.type.452 = facet_value %i64, (%LeftShiftAssignWith.impl_witness.710) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.6b2: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.f8b) [concrete]
// CHECK:STDOUT:   %.305: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.6b2, %LeftShiftAssignWith.facet.f8b [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.c92b6d.1: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.33ee4d.2, @Int.as.LeftShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.c92b6d.2: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.33ee4d.1, @Int.as.LeftShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f17: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.d12507.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.265e34.1: %Int.as.RightShiftAssignWith.impl.Op.type.d12507.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.d12507.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.265e34.2: %Int.as.RightShiftAssignWith.impl.Op.type.d12507.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.90d: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.900, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.ef7750.1: %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.ef7750.2: %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.7a9: %RightShiftAssignWith.type.f17 = facet_value %i64, (%RightShiftAssignWith.impl_witness.90d) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.42e: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.7a9) [concrete]
// CHECK:STDOUT:   %.c58: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.42e, %RightShiftAssignWith.facet.7a9 [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.3e9990.1: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.ef7750.2, @Int.as.RightShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.3e9990.2: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.ef7750.1, @Int.as.RightShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.e95403.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.16c = impl_witness_table (%Core.import_ref.5f1), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8ea: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.e95403.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cc8 = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.63e: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.2 (%Int.as.SubAssignWith.impl.Op.type.b78b27.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.2 (constants.%Int.as.SubAssignWith.impl.Op.e5d778.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.a47 = impl_witness_table (%Core.import_ref.63e), @Int.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.69a: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.1 (%Int.as.SubAssignWith.impl.Op.type.b78b27.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.1 (constants.%Int.as.SubAssignWith.impl.Op.e5d778.2)]
// CHECK:STDOUT:   %Core.import_ref.ec5: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.2 (%Int.as.MulAssignWith.impl.Op.type.dfe9fc.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.2 (constants.%Int.as.MulAssignWith.impl.Op.64e092.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.ef2 = impl_witness_table (%Core.import_ref.ec5), @Int.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.cf1: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.1 (%Int.as.MulAssignWith.impl.Op.type.dfe9fc.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.1 (constants.%Int.as.MulAssignWith.impl.Op.64e092.2)]
// CHECK:STDOUT:   %Core.import_ref.3dc: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.2 (%Int.as.DivAssignWith.impl.Op.type.24ad43.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.2 (constants.%Int.as.DivAssignWith.impl.Op.d4f4ad.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.78c = impl_witness_table (%Core.import_ref.3dc), @Int.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.0bd: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.1 (%Int.as.DivAssignWith.impl.Op.type.24ad43.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.1 (constants.%Int.as.DivAssignWith.impl.Op.d4f4ad.2)]
// CHECK:STDOUT:   %Core.import_ref.e41: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.2 (%Int.as.ModAssignWith.impl.Op.type.c4c096.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.2 (constants.%Int.as.ModAssignWith.impl.Op.568a64.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.8a0 = impl_witness_table (%Core.import_ref.e41), @Int.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.2de: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.1 (%Int.as.ModAssignWith.impl.Op.type.c4c096.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.1 (constants.%Int.as.ModAssignWith.impl.Op.568a64.2)]
// CHECK:STDOUT:   %Core.import_ref.29b: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.2 (%Int.as.BitAndAssignWith.impl.Op.type.b67ba2.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.2 (constants.%Int.as.BitAndAssignWith.impl.Op.af7879.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.c62 = impl_witness_table (%Core.import_ref.29b), @Int.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4e6: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.1 (%Int.as.BitAndAssignWith.impl.Op.type.b67ba2.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.1 (constants.%Int.as.BitAndAssignWith.impl.Op.af7879.2)]
// CHECK:STDOUT:   %Core.import_ref.810e: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.2 (%Int.as.BitOrAssignWith.impl.Op.type.9f45aa.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.2 (constants.%Int.as.BitOrAssignWith.impl.Op.d861a8.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.c22 = impl_witness_table (%Core.import_ref.810e), @Int.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.308: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.1 (%Int.as.BitOrAssignWith.impl.Op.type.9f45aa.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.1 (constants.%Int.as.BitOrAssignWith.impl.Op.d861a8.2)]
// CHECK:STDOUT:   %Core.import_ref.33a1: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.2 (%Int.as.BitXorAssignWith.impl.Op.type.cccbae.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.2 (constants.%Int.as.BitXorAssignWith.impl.Op.ffde62.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.e49 = impl_witness_table (%Core.import_ref.33a1), @Int.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dc6: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.1 (%Int.as.BitXorAssignWith.impl.Op.type.cccbae.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.1 (constants.%Int.as.BitXorAssignWith.impl.Op.ffde62.2)]
// CHECK:STDOUT:   %Core.import_ref.6f8: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.2 (%Int.as.LeftShiftAssignWith.impl.Op.type.42b491.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.2 (constants.%Int.as.LeftShiftAssignWith.impl.Op.dadf31.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.d49 = impl_witness_table (%Core.import_ref.6f8), @Int.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dcf: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.1 (%Int.as.LeftShiftAssignWith.impl.Op.type.42b491.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.1 (constants.%Int.as.LeftShiftAssignWith.impl.Op.dadf31.2)]
// CHECK:STDOUT:   %Core.import_ref.f4d: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.2 (%Int.as.RightShiftAssignWith.impl.Op.type.d12507.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.2 (constants.%Int.as.RightShiftAssignWith.impl.Op.265e34.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.900 = impl_witness_table (%Core.import_ref.f4d), @Int.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b7f: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.1 (%Int.as.RightShiftAssignWith.impl.Op.type.d12507.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.1 (constants.%Int.as.RightShiftAssignWith.impl.Op.265e34.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.34c = impl_witness_access constants.%AddAssignWith.impl_witness.7ac, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = specific_constant imports.%Core.Op.8ea, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.1e9 = impl_witness_access constants.%SubAssignWith.impl_witness.e52, element0 [concrete = constants.%Int.as.SubAssignWith.impl.Op.294900.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Int.as.SubAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.53c9b3.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.SubAssignWith.impl.Op.type.16701f.1 = specific_constant imports.%Core.Op.69a, @Int.as.SubAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.SubAssignWith.impl.Op.294900.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Int.as.SubAssignWith.impl.Op.type.16701f.1 = name_ref Op, %.loc11_5 [concrete = constants.%Int.as.SubAssignWith.impl.Op.294900.1]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Int.as.SubAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.53c9b3.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Int.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.d13 = impl_witness_access constants.%MulAssignWith.impl_witness.289, element0 [concrete = constants.%Int.as.MulAssignWith.impl.Op.c28854.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Int.as.MulAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.f89e08.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.MulAssignWith.impl.Op.type.2a93dc.1 = specific_constant imports.%Core.Op.cf1, @Int.as.MulAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.MulAssignWith.impl.Op.c28854.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Int.as.MulAssignWith.impl.Op.type.2a93dc.1 = name_ref Op, %.loc12_5 [concrete = constants.%Int.as.MulAssignWith.impl.Op.c28854.1]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Int.as.MulAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.f89e08.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Int.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %i64 = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.9e5 = impl_witness_access constants.%DivAssignWith.impl_witness.503, element0 [concrete = constants.%Int.as.DivAssignWith.impl.Op.3415fd.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.DivAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.43b329.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.DivAssignWith.impl.Op.type.cc5ddf.1 = specific_constant imports.%Core.Op.0bd, @Int.as.DivAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.DivAssignWith.impl.Op.3415fd.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Int.as.DivAssignWith.impl.Op.type.cc5ddf.1 = name_ref Op, %.loc13_5 [concrete = constants.%Int.as.DivAssignWith.impl.Op.3415fd.1]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Int.as.DivAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.43b329.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Int.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %i64 = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.300 = impl_witness_access constants.%ModAssignWith.impl_witness.e02, element0 [concrete = constants.%Int.as.ModAssignWith.impl.Op.293f95.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Int.as.ModAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.68356e.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.ModAssignWith.impl.Op.type.977d70.1 = specific_constant imports.%Core.Op.2de, @Int.as.ModAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.ModAssignWith.impl.Op.293f95.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Int.as.ModAssignWith.impl.Op.type.977d70.1 = name_ref Op, %.loc14_5 [concrete = constants.%Int.as.ModAssignWith.impl.Op.293f95.1]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Int.as.ModAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.68356e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Int.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.615 = impl_witness_access constants.%BitAndAssignWith.impl_witness.72d, element0 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.717a47.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Int.as.BitAndAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.fb3f3f.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5: %Int.as.BitAndAssignWith.impl.Op.type.026e1b.1 = specific_constant imports.%Core.Op.4e6, @Int.as.BitAndAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.717a47.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Int.as.BitAndAssignWith.impl.Op.type.026e1b.1 = name_ref Op, %.loc16_5 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.717a47.1]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Int.as.BitAndAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.fb3f3f.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Int.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16: init %i64 = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %i64 = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.491 = impl_witness_access constants.%BitOrAssignWith.impl_witness.6f2, element0 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.9e9473.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Int.as.BitOrAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.741d42.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5: %Int.as.BitOrAssignWith.impl.Op.type.1c2315.1 = specific_constant imports.%Core.Op.308, @Int.as.BitOrAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.9e9473.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Int.as.BitOrAssignWith.impl.Op.type.1c2315.1 = name_ref Op, %.loc17_5 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.9e9473.1]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Int.as.BitOrAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.741d42.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Int.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17: init %i64 = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %i64 = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.3f5 = impl_witness_access constants.%BitXorAssignWith.impl_witness.11a, element0 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.5b03e1.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Int.as.BitXorAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.b395e5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5: %Int.as.BitXorAssignWith.impl.Op.type.02370f.1 = specific_constant imports.%Core.Op.dc6, @Int.as.BitXorAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.5b03e1.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Int.as.BitXorAssignWith.impl.Op.type.02370f.1 = name_ref Op, %.loc18_5 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.5b03e1.1]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Int.as.BitXorAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.b395e5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Int.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18: init %i64 = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %i64 = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.305 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.710, element0 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.33ee4d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Int.as.LeftShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.c92b6d.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5: %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.1 = specific_constant imports.%Core.Op.dcf, @Int.as.LeftShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.33ee4d.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Int.as.LeftShiftAssignWith.impl.Op.type.75984d.1 = name_ref Op, %.loc19_5 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.33ee4d.1]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Int.as.LeftShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.c92b6d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Int.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19: init %i64 = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %i64 = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.c58 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.90d, element0 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.ef7750.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Int.as.RightShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.3e9990.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5: %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.1 = specific_constant imports.%Core.Op.b7f, @Int.as.RightShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.ef7750.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Int.as.RightShiftAssignWith.impl.Op.type.6a669e.1 = name_ref Op, %.loc20_5 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.ef7750.1]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Int.as.RightShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.3e9990.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Int.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.e16: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.2c1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.393661.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.17a680.1: %Cpp.long.as.AddAssignWith.impl.Op.type.393661.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.393661.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.17a680.2: %Cpp.long.as.AddAssignWith.impl.Op.type.393661.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.775: %AddAssignWith.type.5ab = facet_value %Cpp.long, (%AddAssignWith.impl_witness.e16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.77c: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.775) [concrete]
// CHECK:STDOUT:   %.330: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.77c, %AddAssignWith.facet.775 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.a9f447.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.17a680.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.a9f447.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.17a680.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.277 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bc1: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.d3c: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.2c1 = impl_witness_table (%Core.import_ref.d3c), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.373: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1757be.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.1fe1da.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.330 = impl_witness_access constants.%AddAssignWith.impl_witness.e16, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.17a680.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.ecb) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.340 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.a9f447.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.393661.1 = specific_constant imports.%Core.Op.373, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.17a680.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.393661.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.17a680.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.a9f447.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.3(%a.ref, %.loc9_8.2)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_runtime_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.277 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.277 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.77d: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.248: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.38c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.386: %ImplicitAs.type.77d = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.248) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e1f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.386) [concrete]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e1f, %ImplicitAs.facet.386 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a0e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.15f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.ccc: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b83: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.b83) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet.b83 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e06: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.35985e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %U.500: %ImplicitAs.type.35985e.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.1: %Int.as.AddAssignWith.impl.Op.type.e95403.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.2: %Int.as.AddAssignWith.impl.Op.type.e95403.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c56: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cc8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.85c: %ImplicitAs.type.77d = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c56) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.7ac: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.16c, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.35824f.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.1c439d.1: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.35824f.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.1c439d.2: %Int.as.AddAssignWith.impl.Op.type.35824f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.4d3: %AddAssignWith.type.e06 = facet_value %i64, (%AddAssignWith.impl_witness.7ac) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d4a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.4d3) [concrete]
// CHECK:STDOUT:   %.34c: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d4a, %AddAssignWith.facet.4d3 [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.277 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bc1: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.1c439d.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.1c439d.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.85c) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e91, %ImplicitAs.facet.85c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.e95403.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.16c = impl_witness_table (%Core.import_ref.5f1), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8ea: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.e95403.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cc8 = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %i64 = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.71b = impl_witness_access constants.%ImplicitAs.impl_witness.248, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.15f]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a0e]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i64 = call %bound_method.loc8_3.2(%int_1.loc8) [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:   %.loc8: init %i64 = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.ccc]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %a: ref %i64 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.277 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.277 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.34c = impl_witness_access constants.%AddAssignWith.impl_witness.7ac, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = specific_constant imports.%Core.Op.8ea, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.35824f.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.1c439d.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.85c) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3163c0.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.cc6 = impl_witness_access constants.%ImplicitAs.impl_witness.c56, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.ada = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.340: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ecb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.abe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.340 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.6c1: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.067: type = fn_type_with_self_type %Inc.WithSelf.Op.type.6c1, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.9c9: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.964: type = fn_type_with_self_type %Dec.WithSelf.Op.type.9c9, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.277: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.277 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bc1: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.abe = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.71b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.277 = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.277 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bc1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ada = ref_param_pattern [concrete = constants.%self.param_patt.fe4]
// CHECK:STDOUT:     %self.patt: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0d8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.ecb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.067 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.964 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_3.2(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_3.2(%self.param: ref %Cpp.long) {
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.96f: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long.patt: %pattern_type.96f = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b7: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.34c(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.58a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c9b, @Core.IntLiteral.as.ImplicitAs.impl.34c(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.605: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.34c(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6ae: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.605 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.72a: %ImplicitAs.type.6b7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.7ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a9a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.72a) [concrete]
// CHECK:STDOUT:   %.e51: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a9a, %ImplicitAs.facet.72a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b61: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6ae [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6ae, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.024: %u64 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.75d: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.92f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.75d) [concrete]
// CHECK:STDOUT:   %.39a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.92f, %ImplicitAs.facet.75d [concrete]
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
// CHECK:STDOUT:   %carbon_u64.patt: %pattern_type.96f = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   %carbon_u64_result.patt: %pattern_type.97e = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   %f32.789: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.68f: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.940: <witness> = impl_witness imports.%As.impl_witness_table.d4e [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.68f = facet_value Core.IntLiteral, (%As.impl_witness.940) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ef7: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.20a: type = fn_type_with_self_type %As.WithSelf.Convert.type.ef7, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.34b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.090 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.263: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.34b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.eb5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.263) [concrete]
// CHECK:STDOUT:   %.7d5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.eb5, %ImplicitAs.facet.263 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Array.4a1: type = class_type @Array, @Array(%f32.789, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %Array.4a1 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.66c: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.789)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.610, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.55c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e51: %ImplicitAs.type.66c = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.9a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.969: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.789, %ImplicitAs.facet.e51) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.969, %ImplicitAs.facet.e51 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a [concrete]
// CHECK:STDOUT:   %pattern_type.0d4: type = pattern_type %f32.789 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.255: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.9ba: %f32.789 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.4a1 = tuple_value (%float.9ba) [concrete]
// CHECK:STDOUT:   %self.param_patt.b45: %pattern_type.0d4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26b: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt.b45 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.115: %pattern_type.951 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b9: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt.115 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2ef: %pattern_type.97e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a5a: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt.2ef [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.116: %pattern_type.97f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.eb6: %pattern_type.97f = wrapper_binding_pattern self, %self.param_patt.116 [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.f75: @Core.IntLiteral.as.ImplicitAs.impl.34c.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.34c.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.58a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c9b = impl_witness_table (%Core.import_ref.f75), @Core.IntLiteral.as.ImplicitAs.impl.34c [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb8 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.d4e = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.60e [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.090 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long.as.ImplicitAs.impl.8de [concrete]
// CHECK:STDOUT:   %Core.import_ref.f42: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.dd1) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.f4c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.610 = impl_witness_table (%Core.import_ref.f42), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.decl: %ULongLongResult.cpp_destructor.type = fn_decl @ULongLongResult.cpp_destructor [concrete = constants.%ULongLongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ULongResult.cpp_destructor.decl: %ULongResult.cpp_destructor.type = fn_decl @ULongResult.cpp_destructor [concrete = constants.%ULongResult.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc24_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.951 = ref_param_pattern [concrete = constants.%self.param_patt.115]
// CHECK:STDOUT:     %self.patt: %pattern_type.951 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.4a1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.4a1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97e = ref_param_pattern [concrete = constants.%self.param_patt.2ef]
// CHECK:STDOUT:     %self.patt: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ULongLongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ULongLongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc19 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97f = ref_param_pattern [concrete = constants.%self.param_patt.116]
// CHECK:STDOUT:     %self.patt: %pattern_type.97f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eb6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ULongResult = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ULongResult = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.e51 = impl_witness_access constants.%ImplicitAs.impl_witness.7ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6ae]
// CHECK:STDOUT:   %bound_method.loc15_56.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b61]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_56.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.d5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %u64 = call %bound_method.loc15_56.2(%int_1.loc15) [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc15_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc15_56.2: %u64 = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = wrapper_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.96f = value_binding_pattern cpp_unsigned_long_long [concrete = constants.%cpp_unsigned_long_long.patt]
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
// CHECK:STDOUT:   %impl.elem0.loc18: %.39a = impl_witness_access constants.%ImplicitAs.impl_witness.a75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc21: %.e51 = impl_witness_access constants.%ImplicitAs.impl_witness.7ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6ae]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b61]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.d5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %u64.loc21: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_u64: %u64 = wrapper_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.96f = value_binding_pattern carbon_u64 [concrete = constants.%carbon_u64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.4a1 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.dd3 = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e4a]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.255]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.789 = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.789 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.789 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.789 to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.9ba]
// CHECK:STDOUT:   %.loc24_59.5: init %Array.4a1 to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %Array.4a1 = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %Array [concrete = constants.%Array.4a1] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.789 [concrete = constants.%f32.789]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.20a = impl_witness_access constants.%As.impl_witness.940, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.7d5 = impl_witness_access constants.%ImplicitAs.impl_witness.34b, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.789, constants.%int_1.5b8) [concrete = constants.%Array.4a1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.4a1 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.951 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.951 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%a.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_83.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_83.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_93.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_93.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_107.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16(%.loc16_107.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.1(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.1(%self.param: ref %f32.789) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24: ref %f32.789 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.2(%self.param: ref %Array.4a1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.2(%self.param: ref %Array.4a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22(%self.param: ref %ULongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22(%self.param: ref %ULongLongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%ULongLongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc19(%self.param: ref %ULongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc19(%self.param: ref %ULongResult) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call imports.%ULongResult.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.172 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.172 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7da: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7da = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.92f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.39a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.92f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.172 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.172 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2bb: <witness> = impl_witness imports.%Copy.impl_witness_table.2f0 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.2bb) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ca5: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ca5, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.96f: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %self.param_patt.186: %pattern_type.96f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.dfa: %pattern_type.96f = wrapper_binding_pattern self, %self.param_patt.186 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.15d: %pattern_type.172 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1f0: %pattern_type.172 = wrapper_binding_pattern self, %self.param_patt.15d [concrete]
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
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb8 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.a49 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2f0 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.96f = ref_param_pattern [concrete = constants.%self.param_patt.186]
// CHECK:STDOUT:     %self.patt: %pattern_type.96f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.dfa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.96f = ref_param_pattern [concrete = constants.%self.param_patt.186]
// CHECK:STDOUT:     %self.patt: %pattern_type.96f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.dfa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.172 = ref_param_pattern [concrete = constants.%self.param_patt.15d]
// CHECK:STDOUT:     %self.patt: %pattern_type.172 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1f0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.unsigned_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.unsigned_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.172 = ref_param_pattern [concrete = constants.%self.param_patt.15d]
// CHECK:STDOUT:     %self.patt: %pattern_type.172 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1f0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Cpp.unsigned_long = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Cpp.unsigned_long = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.39a = impl_witness_access constants.%ImplicitAs.impl_witness.a75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.63b = impl_witness_access constants.%Copy.impl_witness.2bb, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %Cpp.unsigned_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %Cpp.unsigned_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %Cpp.unsigned_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
