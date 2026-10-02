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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %carbon_long.patt: %pattern_type.6d9 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.025: %ImplicitAs.type.270 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47a, %ImplicitAs.facet.025 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.006: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.f0a: <witness> = impl_witness imports.%As.impl_witness_table.9e8 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.006 = facet_value Core.IntLiteral, (%As.impl_witness.f0a) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.57c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.450: type = fn_type_with_self_type %As.WithSelf.Convert.type.57c, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.88a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d5b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e81: %ImplicitAs.type.5be = facet_value %Cpp.long, (%ImplicitAs.impl_witness.88a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.23e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.e81) [concrete]
// CHECK:STDOUT:   %.db0: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.23e, %ImplicitAs.facet.e81 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.33f, %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.825 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.9e8 = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.c18 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d5b = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.long.as.ImplicitAs.impl.ee4 [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9(%cpp_long.ref)
// CHECK:STDOUT:   %.loc9_33.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_33.2: %i64 = converted %cpp_long.ref, %.loc9_33.1
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = wrapper_binding carbon_long, %.loc9_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.6d9 = value_binding_pattern carbon_long [concrete = constants.%carbon_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
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
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.450 = impl_witness_access constants.%As.impl_witness.f0a, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.db0 = impl_witness_access constants.%ImplicitAs.impl_witness.88a, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long.patt: %pattern_type.172 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.62c: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.615: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1bb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.444 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8d1: %ImplicitAs.type.62c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1bb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.23a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.8d1) [concrete]
// CHECK:STDOUT:   %.90c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.23a, %ImplicitAs.facet.8d1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_long.patt: %pattern_type.878 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c49: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8bb [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e0c: %ImplicitAs.type.615 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.c49) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d5f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.e0c) [concrete]
// CHECK:STDOUT:   %.cfb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d5f, %ImplicitAs.facet.e0c [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.1b7: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.4c4: <witness> = impl_witness imports.%As.impl_witness_table.725 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.1b7 = facet_value Core.IntLiteral, (%As.impl_witness.4c4) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f25: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.a36: type = fn_type_with_self_type %As.WithSelf.Convert.type.f25, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.db9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d28 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4a1: %ImplicitAs.type.5be = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.db9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.10e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.4a1) [concrete]
// CHECK:STDOUT:   %.bfb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.10e, %ImplicitAs.facet.4a1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
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
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.444 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.716 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8bb = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.unsigned_long.as.ImplicitAs.impl.0e2 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.725 = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.As.impl.ac7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.4: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d28 = impl_witness_table (%Core.import_ref.06c923.4), @Cpp.unsigned_long.as.ImplicitAs.impl.b87 [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.90c = impl_witness_access constants.%ImplicitAs.impl_witness.1bb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.cfb = impl_witness_access constants.%ImplicitAs.impl_witness.c49, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %cpp_unsigned_long.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9: init %u64 = call %bound_method.loc9(%cpp_unsigned_long.ref)
// CHECK:STDOUT:   %.loc9_42.1: %u64 = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_42.2: %u64 = converted %cpp_unsigned_long.ref, %.loc9_42.1
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = wrapper_binding carbon_unsigned_long, %.loc9_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.878 = value_binding_pattern carbon_unsigned_long [concrete = constants.%carbon_unsigned_long.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
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
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.a36 = impl_witness_access constants.%As.impl_witness.4c4, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc11_30.1(%int_1.loc11) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc11_30.2: %Cpp.unsigned_long = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.bfb = impl_witness_access constants.%ImplicitAs.impl_witness.db9, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11: init Core.IntLiteral = call %bound_method.loc11_30.2(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %cpp_long_long.patt: %pattern_type.6d9 = value_binding_pattern cpp_long_long [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.013: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.4f6: %i64 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
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
// CHECK:STDOUT:   %carbon_i64.patt: %pattern_type.6d9 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   %carbon_i64_result.patt: %pattern_type.357 = value_binding_pattern carbon_i64_result [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.19a = impl_witness_access constants.%ImplicitAs.impl_witness.122, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.64f]
// CHECK:STDOUT:   %bound_method.loc15_38.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_38.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.013]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_38.2(%int_1.loc15) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc15_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc15_38.2: %i64 = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = wrapper_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.6d9 = value_binding_pattern cpp_long_long [concrete = constants.%cpp_long_long.patt]
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
// CHECK:STDOUT:   %impl.elem0.loc18: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.6d9 = value_binding_pattern carbon_i64 [concrete = constants.%carbon_i64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24_55: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn.loc24_55 [concrete = constants.%bound_method.29b]
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
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.e74 = impl_witness_access constants.%As.impl_witness.291, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.20c]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.1: <specific function> = specific_function %impl.elem0.loc24_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %int_1.loc24, %specific_fn.loc24_30.1 [concrete = constants.%bound_method.29d]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc24_30.2(%int_1.loc24) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %.loc24_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %.loc24_30.2: %i64 = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.8f5 = impl_witness_access constants.%ImplicitAs.impl_witness.8f7, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.33b]
// CHECK:STDOUT:     %bound_method.loc24_30.3: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc24_30.2: <specific function> = specific_function %impl.elem0.loc24_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_30.4: <bound method> = bound_method %.loc24_30.2, %specific_fn.loc24_30.2 [concrete = constants.%bound_method.09e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.4(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_81.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_81.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_73.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_73.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_87.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.78e: <witness> = impl_witness imports.%Copy.impl_witness_table.842 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.78e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.200: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3ca: type = fn_type_with_self_type %Copy.WithSelf.Op.type.200, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.842 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.3ca = impl_witness_access constants.%Copy.impl_witness.78e, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %EqWith.type.006: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.b53: <witness> = impl_witness imports.%EqWith.impl_witness_table.c27 [concrete]
// CHECK:STDOUT:   %EqWith.facet.225: %EqWith.type.006 = facet_value %Cpp.long, (%EqWith.impl_witness.b53) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.933: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.225) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.f91: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.225) [concrete]
// CHECK:STDOUT:   %.c26: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.933, %EqWith.facet.225 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1ca: %Cpp.long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.0d5: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.f91, %EqWith.facet.225 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a6a: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b27: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.b84: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.303 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.f2b: %OrderedWith.type.b27 = facet_value %Cpp.long, (%OrderedWith.impl_witness.b84) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.2f1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.f2b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.61d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.f2b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ab1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.f2b) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.f2b) [concrete]
// CHECK:STDOUT:   %.ce1: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ab1, %OrderedWith.facet.f2b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.7c8: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.412: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.2f1, %OrderedWith.facet.f2b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.606: %Cpp.long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.1f5: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd1, %OrderedWith.facet.f2b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.fc6: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.61d, %OrderedWith.facet.f2b [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a874: %Cpp.long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.c27 = impl_witness_table (%Core.import_ref.a874, %Core.import_ref.adc), @Cpp.long.as.EqWith.impl.8aa [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e88: %Cpp.long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.303 = impl_witness_table (%Core.import_ref.0e88, %Core.import_ref.757, %Core.import_ref.c32, %Core.import_ref.381), @Cpp.long.as.OrderedWith.impl.f3c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.c26 = impl_witness_access constants.%EqWith.impl_witness.b53, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.0d5 = impl_witness_access constants.%EqWith.impl_witness.b53, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.ce1 = impl_witness_access constants.%OrderedWith.impl_witness.b84, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.412 = impl_witness_access constants.%OrderedWith.impl_witness.b84, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.1f5 = impl_witness_access constants.%OrderedWith.impl_witness.b84, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.fc6 = impl_witness_access constants.%OrderedWith.impl_witness.b84, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
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
// CHECK:STDOUT:   %Int.ec967b.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.3c0: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.2ef: %ImplicitAs.type.3c9 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.c5a(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.ba1dc6.1: %Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.0deb16.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.c5a(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.f0198d.1: %Cpp.long.as.EqWith.impl.Equal.type.0deb16.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.c5a(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.ba1dc6.2: %Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.0deb16.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.c5a(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.f0198d.2: %Cpp.long.as.EqWith.impl.Equal.type.0deb16.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3758b9.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.ec967b.1)> [symbolic]
// CHECK:STDOUT:   %T.722: %ImplicitAs.type.3758b9.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.37a(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.803d3c.1: %T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.ecab8f.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.37a(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f04247.1: %T.as_type.as.EqWith.impl.Equal.type.ecab8f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.37a(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.803d3c.2: %T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.ecab8f.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.37a(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f04247.2: %T.as_type.as.EqWith.impl.Equal.type.ecab8f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.61f: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.025: %ImplicitAs.type.270 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.e40: <witness> = impl_witness imports.%EqWith.impl_witness_table.bb4, @T.as_type.as.EqWith.impl.37a(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.230367.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.8, @T.as_type.as.EqWith.impl.37a(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c8179c.1: %T.as_type.as.EqWith.impl.Equal.type.230367.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.74746f.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.8, @T.as_type.as.EqWith.impl.37a(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.07b057.1: %T.as_type.as.EqWith.impl.NotEqual.type.74746f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.230367.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.7, @T.as_type.as.EqWith.impl.37a(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.c8179c.2: %T.as_type.as.EqWith.impl.Equal.type.230367.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.74746f.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.7, @T.as_type.as.EqWith.impl.37a(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.07b057.2: %T.as_type.as.EqWith.impl.NotEqual.type.74746f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.076: %EqWith.type.3c0 = facet_value %Cpp.long, (%EqWith.impl_witness.e40) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.221: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.076) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.201: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.076) [concrete]
// CHECK:STDOUT:   %.35a: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.221, %EqWith.facet.076 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.02e8cd.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c8179c.2, @T.as_type.as.EqWith.impl.Equal.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.02e8cd.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.c8179c.1, @T.as_type.as.EqWith.impl.Equal.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47a, %ImplicitAs.facet.025 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.19e: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.201, %EqWith.facet.076 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.596c98.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.07b057.2, @T.as_type.as.EqWith.impl.NotEqual.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.596c98.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.07b057.1, @T.as_type.as.EqWith.impl.NotEqual.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.914: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.18d2b9.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.785fc2.1: %Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2aa95c.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.e7708d.1: %Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.18d2b9.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.785fc2.2: %Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2aa95c.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.344(%T.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.e7708d.2: %Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.667429.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.73c811.1: %T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5b43c0.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.bdd39a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e25dd1.1: %T.as_type.as.OrderedWith.impl.Less.type.bdd39a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.667429.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.73c811.2: %T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.5b43c0.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.bdd39a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.d50(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e25dd1.2: %T.as_type.as.OrderedWith.impl.Less.type.bdd39a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.024: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.600: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.e70, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.f22853.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.8, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.524a84.1: %T.as_type.as.OrderedWith.impl.Less.type.f22853.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.8, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.adef8c.1: %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.f22853.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.7, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.524a84.2: %T.as_type.as.OrderedWith.impl.Less.type.f22853.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.7, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.adef8c.2: %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7, @T.as_type.as.OrderedWith.impl.d50(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.de4: %OrderedWith.type.914 = facet_value %Cpp.long, (%OrderedWith.impl_witness.600) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f7b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.de4) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.20f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.de4) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.29f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.de4) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ce9: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.de4) [concrete]
// CHECK:STDOUT:   %.8d4: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.29f, %OrderedWith.facet.de4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4f98eb.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.adef8c.2, @T.as_type.as.OrderedWith.impl.Greater.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4f98eb.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.adef8c.1, @T.as_type.as.OrderedWith.impl.Greater.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.704: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f7b, %OrderedWith.facet.de4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.f08135.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.524a84.2, @T.as_type.as.OrderedWith.impl.Less.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.f08135.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.524a84.1, @T.as_type.as.OrderedWith.impl.Less.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.410: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ce9, %OrderedWith.facet.de4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2315a3.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2315a3.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.f33: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.20f, %OrderedWith.facet.de4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19a3b3.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19a3b3.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.570: <witness> = impl_witness imports.%EqWith.impl_witness_table.537, @Cpp.long.as.EqWith.impl.c5a(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.096933.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.c5a(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.e36cb8.1: %Cpp.long.as.EqWith.impl.Equal.type.096933.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.c5a(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6880e5.1: %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.096933.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.c5a(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.e36cb8.2: %Cpp.long.as.EqWith.impl.Equal.type.096933.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.c5a(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6880e5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.c2d: %EqWith.type.61f = facet_value %Cpp.long, (%EqWith.impl_witness.570) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.bc2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.926: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.c2d) [concrete]
// CHECK:STDOUT:   %.6dc: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.bc2, %EqWith.facet.c2d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.27fe38.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.e36cb8.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.27fe38.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.e36cb8.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.092: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.926, %EqWith.facet.c2d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.b9a91f.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6880e5.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.b9a91f.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6880e5.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.011: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.68f, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.5a83e8.1: %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.893f0a.1: %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.5a83e8.2: %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.893f0a.2: %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.344(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.cab: %OrderedWith.type.024 = facet_value %Cpp.long, (%OrderedWith.impl_witness.011) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.048: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.996: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.b4c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.1c5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.cab) [concrete]
// CHECK:STDOUT:   %.e92: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.b4c, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.6615e0.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.893f0a.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.6615e0.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.893f0a.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.9c1: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.048, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.d570af.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5a83e8.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.d570af.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5a83e8.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.519: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.1c5, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7986d9.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7986d9.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.942: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.996, %OrderedWith.facet.cab [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b4c9be.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b4c9be.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c7e: @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.0deb16.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.f0198d.1)]
// CHECK:STDOUT:   %Core.import_ref.7ec: @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.ba1dc6.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.537 = impl_witness_table (%Core.import_ref.c7e, %Core.import_ref.7ec), @Cpp.long.as.EqWith.impl.c5a [concrete]
// CHECK:STDOUT:   %Core.Equal.4b2: @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.0deb16.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.f0198d.2)]
// CHECK:STDOUT:   %Core.NotEqual.a67: @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.f949cd.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.c5a.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.ba1dc6.2)]
// CHECK:STDOUT:   %Core.import_ref.241e: @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.ecab8f.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.f04247.1)]
// CHECK:STDOUT:   %Core.import_ref.ecb: @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.803d3c.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.bb4 = impl_witness_table (%Core.import_ref.241e, %Core.import_ref.ecb), @T.as_type.as.EqWith.impl.37a [concrete]
// CHECK:STDOUT:   %Core.Equal.7f1: @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.ecab8f.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.f04247.2)]
// CHECK:STDOUT:   %Core.NotEqual.468: @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.3ee15c.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.37a.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.803d3c.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.825 [concrete]
// CHECK:STDOUT:   %Core.import_ref.550d: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.e7708d.1)]
// CHECK:STDOUT:   %Core.import_ref.8db: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2aa95c.1)]
// CHECK:STDOUT:   %Core.import_ref.43a: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.785fc2.1)]
// CHECK:STDOUT:   %Core.import_ref.f99: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.18d2b9.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.68f = impl_witness_table (%Core.import_ref.550d, %Core.import_ref.8db, %Core.import_ref.43a, %Core.import_ref.f99), @Cpp.long.as.OrderedWith.impl.344 [concrete]
// CHECK:STDOUT:   %Core.Less.8b8: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.cde7b1.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.e7708d.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.9be: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.3b0145.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2aa95c.2)]
// CHECK:STDOUT:   %Core.Greater.5f1: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.6700ed.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.785fc2.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.606: @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b4e667.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.344.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.18d2b9.2)]
// CHECK:STDOUT:   %Core.import_ref.251: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.bdd39a.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.e25dd1.1)]
// CHECK:STDOUT:   %Core.import_ref.b5e: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.5b43c0.1)]
// CHECK:STDOUT:   %Core.import_ref.d2c: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.73c811.1)]
// CHECK:STDOUT:   %Core.import_ref.755: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.667429.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.e70 = impl_witness_table (%Core.import_ref.251, %Core.import_ref.b5e, %Core.import_ref.d2c, %Core.import_ref.755), @T.as_type.as.OrderedWith.impl.d50 [concrete]
// CHECK:STDOUT:   %Core.Less.b39: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.bdd39a.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.e25dd1.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.1fb: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.d98ceb.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.5b43c0.2)]
// CHECK:STDOUT:   %Core.Greater.022: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.cde0ef.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.73c811.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.91d: @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.7f1b37.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d50.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.667429.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.35a = impl_witness_access constants.%EqWith.impl_witness.e40, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c8179c.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.02e8cd.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.230367.1 = specific_constant imports.%Core.Equal.7f1, @T.as_type.as.EqWith.impl.37a(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c8179c.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.230367.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.c8179c.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.02e8cd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_3(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %i64 = converted %a.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.19e = impl_witness_access constants.%EqWith.impl_witness.e40, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.07b057.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.596c98.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.74746f.1 = specific_constant imports.%Core.NotEqual.468, @T.as_type.as.EqWith.impl.37a(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.07b057.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.74746f.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.07b057.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.596c98.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_3(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %i64 = converted %a.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.8d4 = impl_witness_access constants.%OrderedWith.impl_witness.600, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.adef8c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4f98eb.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.1 = specific_constant imports.%Core.Greater.022, @T.as_type.as.OrderedWith.impl.d50(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.adef8c.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.c668ac.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.adef8c.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4f98eb.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_3(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %i64 = converted %a.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.704 = impl_witness_access constants.%OrderedWith.impl_witness.600, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.524a84.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.f08135.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.f22853.1 = specific_constant imports.%Core.Less.b39, @T.as_type.as.OrderedWith.impl.d50(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.524a84.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.f22853.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.524a84.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.f08135.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_3(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %i64 = converted %a.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.410 = impl_witness_access constants.%OrderedWith.impl_witness.600, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2315a3.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.1 = specific_constant imports.%Core.GreaterOrEquivalent.91d, @T.as_type.as.OrderedWith.impl.d50(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c0557.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d68d49.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.2315a3.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_3(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %i64 = converted %a.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.f33 = impl_witness_access constants.%OrderedWith.impl_witness.600, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.7(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19a3b3.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.1 = specific_constant imports.%Core.LessOrEquivalent.1fb, @T.as_type.as.OrderedWith.impl.d50(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.db2672.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.f80dc5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.8(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.19a3b3.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_3(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %i64 = converted %a.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %b.ref.loc15)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.6dc = impl_witness_access constants.%EqWith.impl_witness.570, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e36cb8.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.27fe38.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.EqWith.impl.Equal.type.096933.1 = specific_constant imports.%Core.Equal.4b2, @Cpp.long.as.EqWith.impl.c5a(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e36cb8.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.096933.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e36cb8.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.27fe38.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.092 = impl_witness_access constants.%EqWith.impl_witness.570, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6880e5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.b9a91f.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.1 = specific_constant imports.%Core.NotEqual.a67, @Cpp.long.as.EqWith.impl.c5a(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6880e5.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.06049f.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6880e5.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.b9a91f.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.e92 = impl_witness_access constants.%OrderedWith.impl_witness.011, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.893f0a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.6615e0.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.1 = specific_constant imports.%Core.Greater.5f1, @Cpp.long.as.OrderedWith.impl.344(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.893f0a.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.7191f2.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.893f0a.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.6615e0.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.9c1 = impl_witness_access constants.%OrderedWith.impl_witness.011, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5a83e8.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.d570af.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.1 = specific_constant imports.%Core.Less.8b8, @Cpp.long.as.OrderedWith.impl.344(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5a83e8.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.f9c766.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5a83e8.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.d570af.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.519 = impl_witness_access constants.%OrderedWith.impl_witness.011, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7986d9.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.1 = specific_constant imports.%Core.GreaterOrEquivalent.606, @Cpp.long.as.OrderedWith.impl.344(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.477f81.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.e33bf0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.7986d9.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc21: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.942 = impl_witness_access constants.%OrderedWith.impl_witness.011, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b4c9be.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.1 = specific_constant imports.%Core.LessOrEquivalent.9be, @Cpp.long.as.OrderedWith.impl.344(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.5f6d89.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.daa7e2.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.ee1) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b4c9be.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc22: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %Int.ec967b.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.006: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.2ef: %ImplicitAs.type.3c9 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.18941e.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.f28(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.989849.1: %T.as_type.as.EqWith.impl.NotEqual.type.18941e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.d4d6e0.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.f28(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.b6e461.1: %T.as_type.as.EqWith.impl.Equal.type.d4d6e0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.18941e.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.f28(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.989849.2: %T.as_type.as.EqWith.impl.NotEqual.type.18941e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.d4d6e0.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.f28(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.b6e461.2: %T.as_type.as.EqWith.impl.Equal.type.d4d6e0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3758b9.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.ec967b.1)> [symbolic]
// CHECK:STDOUT:   %T.722: %ImplicitAs.type.3758b9.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.f429ff.1: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.2c6(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.2d9180.1: %Int.as.EqWith.impl.NotEqual.type.f429ff.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.a75f4c.1: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.2c6(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.794f09.1: %Int.as.EqWith.impl.Equal.type.a75f4c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.f429ff.2: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.2c6(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.2d9180.2: %Int.as.EqWith.impl.NotEqual.type.f429ff.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.a75f4c.2: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.2c6(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.794f09.2: %Int.as.EqWith.impl.Equal.type.a75f4c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.025: %ImplicitAs.type.270 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.6af: <witness> = impl_witness imports.%EqWith.impl_witness_table.430, @Int.as.EqWith.impl.2c6(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.16a85b.1: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.2c6(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.d93e03.1: %Int.as.EqWith.impl.Equal.type.16a85b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.10d9c1.1: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.2c6(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.34a436.1: %Int.as.EqWith.impl.NotEqual.type.10d9c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.16a85b.2: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.2c6(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.d93e03.2: %Int.as.EqWith.impl.Equal.type.16a85b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.10d9c1.2: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.2c6(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.34a436.2: %Int.as.EqWith.impl.NotEqual.type.10d9c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.a37: %EqWith.type.006 = facet_value %i64, (%EqWith.impl_witness.6af) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.458: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.a37) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.2a8: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.a37) [concrete]
// CHECK:STDOUT:   %.440: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.458, %EqWith.facet.a37 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.e64797.1: <specific function> = specific_function %Int.as.EqWith.impl.Equal.d93e03.2, @Int.as.EqWith.impl.Equal.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.e64797.2: <specific function> = specific_function %Int.as.EqWith.impl.Equal.d93e03.1, @Int.as.EqWith.impl.Equal.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47a, %ImplicitAs.facet.025 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.52f: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.2a8, %EqWith.facet.a37 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.d0e55e.1: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.34a436.2, @Int.as.EqWith.impl.NotEqual.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.d0e55e.2: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.34a436.1, @Int.as.EqWith.impl.NotEqual.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b27: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.e4c7ad.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b33f56.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d3f113.1: %T.as_type.as.OrderedWith.impl.Greater.type.b33f56.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.11fbc3.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.e0107f.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.474415.1: %T.as_type.as.OrderedWith.impl.Less.type.e0107f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.e4c7ad.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b33f56.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d3f113.2: %T.as_type.as.OrderedWith.impl.Greater.type.b33f56.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.11fbc3.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.e0107f.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.804(%T.2ef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.474415.2: %T.as_type.as.OrderedWith.impl.Less.type.e0107f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.871b13.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.48295c.1: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.0c8ee8.1: %Int.as.OrderedWith.impl.Greater.type.48295c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.17481f.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.cda090.1: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.ec238a.1: %Int.as.OrderedWith.impl.Less.type.cda090.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.871b13.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.48295c.2: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.0c8ee8.2: %Int.as.OrderedWith.impl.Greater.type.48295c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.17481f.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.cda090.2: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.4e9(%N, %T.722) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.ec238a.2: %Int.as.OrderedWith.impl.Less.type.cda090.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ddc: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.609, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.1b126d.1: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.d7e9c1.1: %Int.as.OrderedWith.impl.Less.type.1b126d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.0ff348.1: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.82d09e.1: %Int.as.OrderedWith.impl.Greater.type.0ff348.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.1b126d.2: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.d7e9c1.2: %Int.as.OrderedWith.impl.Less.type.1b126d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.0ff348.2: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.82d09e.2: %Int.as.OrderedWith.impl.Greater.type.0ff348.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.4e9(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.2e6: %OrderedWith.type.b27 = facet_value %i64, (%OrderedWith.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.837: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2e6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.891: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2e6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.f91: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2e6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.66d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2e6) [concrete]
// CHECK:STDOUT:   %.054: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.f91, %OrderedWith.facet.2e6 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.2c834b.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.82d09e.2, @Int.as.OrderedWith.impl.Greater.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.2c834b.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.82d09e.1, @Int.as.OrderedWith.impl.Greater.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.cb0: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.837, %OrderedWith.facet.2e6 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.9e1a9c.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.d7e9c1.2, @Int.as.OrderedWith.impl.Less.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.9e1a9c.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.d7e9c1.1, @Int.as.OrderedWith.impl.Less.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.22e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.66d, %OrderedWith.facet.2e6 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.18f476.1: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.2, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.18f476.2: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.1, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.873: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.891, %OrderedWith.facet.2e6 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb588a.1: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.2, @Int.as.OrderedWith.impl.LessOrEquivalent.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb588a.2: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.1, @Int.as.OrderedWith.impl.LessOrEquivalent.3(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.e5c: <witness> = impl_witness imports.%EqWith.impl_witness_table.3fe, @T.as_type.as.EqWith.impl.f28(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.abbd95.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.f28(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.4052fe.1: %T.as_type.as.EqWith.impl.Equal.type.abbd95.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.f28(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.39a331.1: %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.abbd95.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.f28(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.4052fe.2: %T.as_type.as.EqWith.impl.Equal.type.abbd95.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.f28(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.39a331.2: %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.bd0: %EqWith.type.006 = facet_value Core.IntLiteral, (%EqWith.impl_witness.e5c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.32b: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.bd0) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.273: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.bd0) [concrete]
// CHECK:STDOUT:   %.b552: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.32b, %EqWith.facet.bd0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.33f085.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.4052fe.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.4052fe.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.68609a.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.33f085.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.4052fe.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.4052fe.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.68609a.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.2 [concrete]
// CHECK:STDOUT:   %.503: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.273, %EqWith.facet.bd0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.e1699c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.39a331.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.39a331.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.420b98.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.e1699c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.39a331.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.39a331.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.420b98.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2de: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.27a, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.b8de4c.1: %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d2135f.1: %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.b8de4c.2: %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.d2135f.2: %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.804(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.230: %OrderedWith.type.b27 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.2de) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.5f4: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.230) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.57f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.230) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.517: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.230) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.258: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.230) [concrete]
// CHECK:STDOUT:   %.491: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.517, %OrderedWith.facet.230 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.ab042e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.d2135f.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.d2135f.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.e6476b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.ab042e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.d2135f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.d2135f.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.e6476b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.2 [concrete]
// CHECK:STDOUT:   %.906: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.5f4, %OrderedWith.facet.230 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.621d39.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.b8de4c.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.b8de4c.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.c40960.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.621d39.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.b8de4c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.b8de4c.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.c40960.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.2 [concrete]
// CHECK:STDOUT:   %.539c: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.258, %OrderedWith.facet.230 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.308b69.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.b1f5e5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.308b69.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.b1f5e5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.2 [concrete]
// CHECK:STDOUT:   %.061: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.57f, %OrderedWith.facet.230 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.e3b7a4.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.7a8c59.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.e3b7a4.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %bound_method.7a8c59.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.163: @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.d4d6e0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.b6e461.1)]
// CHECK:STDOUT:   %Core.import_ref.d18: @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.18941e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.989849.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.163, %Core.import_ref.d18), @T.as_type.as.EqWith.impl.f28 [concrete]
// CHECK:STDOUT:   %Core.Equal.2a7: @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.d4d6e0.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.b6e461.2)]
// CHECK:STDOUT:   %Core.NotEqual.d53: @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.18941e.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.f28.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.989849.2)]
// CHECK:STDOUT:   %Core.import_ref.f64: @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.Equal.type.2 (%Int.as.EqWith.impl.Equal.type.a75f4c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.Equal.2 (constants.%Int.as.EqWith.impl.Equal.794f09.1)]
// CHECK:STDOUT:   %Core.import_ref.34b: @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.NotEqual.type.2 (%Int.as.EqWith.impl.NotEqual.type.f429ff.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.NotEqual.2 (constants.%Int.as.EqWith.impl.NotEqual.2d9180.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.430 = impl_witness_table (%Core.import_ref.f64, %Core.import_ref.34b), @Int.as.EqWith.impl.2c6 [concrete]
// CHECK:STDOUT:   %Core.Equal.342: @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.Equal.type.1 (%Int.as.EqWith.impl.Equal.type.a75f4c.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.Equal.1 (constants.%Int.as.EqWith.impl.Equal.794f09.2)]
// CHECK:STDOUT:   %Core.NotEqual.e8f: @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.NotEqual.type.1 (%Int.as.EqWith.impl.NotEqual.type.f429ff.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @Int.as.EqWith.impl.2c6.%Int.as.EqWith.impl.NotEqual.1 (constants.%Int.as.EqWith.impl.NotEqual.2d9180.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.825 [concrete]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.e0107f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.474415.1)]
// CHECK:STDOUT:   %Core.import_ref.d066: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.11fbc3.1)]
// CHECK:STDOUT:   %Core.import_ref.a55: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.b33f56.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.d3f113.1)]
// CHECK:STDOUT:   %Core.import_ref.1cf: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.e4c7ad.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.27a = impl_witness_table (%Core.import_ref.af7, %Core.import_ref.d066, %Core.import_ref.a55, %Core.import_ref.1cf), @T.as_type.as.OrderedWith.impl.804 [concrete]
// CHECK:STDOUT:   %Core.Less.bb5: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.e0107f.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.474415.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.d62: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.78e5ec.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.11fbc3.2)]
// CHECK:STDOUT:   %Core.Greater.c22: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.b33f56.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.d3f113.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.6db: @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.f321ee.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.804.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.e4c7ad.2)]
// CHECK:STDOUT:   %Core.import_ref.8f6: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Less.type.2 (%Int.as.OrderedWith.impl.Less.type.cda090.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Less.2 (constants.%Int.as.OrderedWith.impl.Less.ec238a.1)]
// CHECK:STDOUT:   %Core.import_ref.1de65: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.17481f.1)]
// CHECK:STDOUT:   %Core.import_ref.c497: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Greater.type.2 (%Int.as.OrderedWith.impl.Greater.type.48295c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Greater.2 (constants.%Int.as.OrderedWith.impl.Greater.0c8ee8.1)]
// CHECK:STDOUT:   %Core.import_ref.095: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.871b13.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.609 = impl_witness_table (%Core.import_ref.8f6, %Core.import_ref.1de65, %Core.import_ref.c497, %Core.import_ref.095), @Int.as.OrderedWith.impl.4e9 [concrete]
// CHECK:STDOUT:   %Core.Less.38a: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Less.type.1 (%Int.as.OrderedWith.impl.Less.type.cda090.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Less.1 (constants.%Int.as.OrderedWith.impl.Less.ec238a.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.c86: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.186452.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.17481f.2)]
// CHECK:STDOUT:   %Core.Greater.429: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Greater.type.1 (%Int.as.OrderedWith.impl.Greater.type.48295c.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.Greater.1 (constants.%Int.as.OrderedWith.impl.Greater.0c8ee8.2)]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.579: @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.20815e.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.4e9.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.871b13.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.440 = impl_witness_access constants.%EqWith.impl_witness.6af, element0 [concrete = constants.%Int.as.EqWith.impl.Equal.d93e03.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.EqWith.impl.Equal.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.e64797.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.EqWith.impl.Equal.type.16a85b.1 = specific_constant imports.%Core.Equal.342, @Int.as.EqWith.impl.2c6(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.Equal.d93e03.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Int.as.EqWith.impl.Equal.type.16a85b.1 = name_ref Equal, %.loc10_5 [concrete = constants.%Int.as.EqWith.impl.Equal.d93e03.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc10, @Int.as.EqWith.impl.Equal.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.e64797.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %Int.as.EqWith.impl.Equal.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %a.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10_5.3(%b.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.52f = impl_witness_access constants.%EqWith.impl_witness.6af, element1 [concrete = constants.%Int.as.EqWith.impl.NotEqual.34a436.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Int.as.EqWith.impl.NotEqual.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.d0e55e.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.EqWith.impl.NotEqual.type.10d9c1.1 = specific_constant imports.%Core.NotEqual.e8f, @Int.as.EqWith.impl.2c6(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.NotEqual.34a436.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Int.as.EqWith.impl.NotEqual.type.10d9c1.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%Int.as.EqWith.impl.NotEqual.34a436.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc11, @Int.as.EqWith.impl.NotEqual.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.d0e55e.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %Int.as.EqWith.impl.NotEqual.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %a.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11_5.3(%b.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.054 = impl_witness_access constants.%OrderedWith.impl_witness.ddc, element2 [concrete = constants.%Int.as.OrderedWith.impl.Greater.82d09e.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Int.as.OrderedWith.impl.Greater.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.2c834b.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.OrderedWith.impl.Greater.type.0ff348.1 = specific_constant imports.%Core.Greater.429, @Int.as.OrderedWith.impl.4e9(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Greater.82d09e.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Int.as.OrderedWith.impl.Greater.type.0ff348.1 = name_ref Greater, %.loc12_5 [concrete = constants.%Int.as.OrderedWith.impl.Greater.82d09e.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc12, @Int.as.OrderedWith.impl.Greater.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.2c834b.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %Int.as.OrderedWith.impl.Greater.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_7(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %i64 = converted %a.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12_5.3(%b.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.cb0 = impl_witness_access constants.%OrderedWith.impl_witness.ddc, element0 [concrete = constants.%Int.as.OrderedWith.impl.Less.d7e9c1.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.OrderedWith.impl.Less.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.9e1a9c.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.OrderedWith.impl.Less.type.1b126d.1 = specific_constant imports.%Core.Less.38a, @Int.as.OrderedWith.impl.4e9(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Less.d7e9c1.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Int.as.OrderedWith.impl.Less.type.1b126d.1 = name_ref Less, %.loc13_5 [concrete = constants.%Int.as.OrderedWith.impl.Less.d7e9c1.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc13, @Int.as.OrderedWith.impl.Less.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.9e1a9c.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %Int.as.OrderedWith.impl.Less.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_7(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %i64 = converted %a.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_5.3(%b.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.22e = impl_witness_access constants.%OrderedWith.impl_witness.ddc, element3 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.18f476.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.1 = specific_constant imports.%Core.GreaterOrEquivalent.579, @Int.as.OrderedWith.impl.4e9(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.989585.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.96037e.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.18f476.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %a.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14_5.3(%b.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.873 = impl_witness_access constants.%OrderedWith.impl_witness.ddc, element1 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb588a.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.1 = specific_constant imports.%Core.LessOrEquivalent.c86, @Int.as.OrderedWith.impl.4e9(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Int.as.OrderedWith.impl.LessOrEquivalent.type.0244f1.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.28ca38.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Int.as.OrderedWith.impl.LessOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bb588a.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc15: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_8(%a.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %i64 = converted %a.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15_5.3(%b.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.b552 = impl_witness_access constants.%EqWith.impl_witness.e5c, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.4052fe.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.33f085.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.68609a.1]
// CHECK:STDOUT:   %.loc17_5.3: %T.as_type.as.EqWith.impl.Equal.type.abbd95.1 = specific_constant imports.%Core.Equal.2a7, @T.as_type.as.EqWith.impl.f28(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.4052fe.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.abbd95.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.4052fe.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.33f085.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.12e9c3.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn [concrete = constants.%bound_method.68609a.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.503 = impl_witness_access constants.%EqWith.impl_witness.e5c, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.39a331.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.e1699c.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.420b98.1]
// CHECK:STDOUT:   %.loc18_5.3: %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.1 = specific_constant imports.%Core.NotEqual.d53, @T.as_type.as.EqWith.impl.f28(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.39a331.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.6cb43d.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.39a331.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.e1699c.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.672322.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn [concrete = constants.%bound_method.420b98.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.491 = impl_witness_access constants.%OrderedWith.impl_witness.2de, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.d2135f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.ab042e.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.e6476b.1]
// CHECK:STDOUT:   %.loc19_5.3: %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.1 = specific_constant imports.%Core.Greater.c22, @T.as_type.as.OrderedWith.impl.804(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.d2135f.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.b0cc60.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.d2135f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.ab042e.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.74a733.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn [concrete = constants.%bound_method.e6476b.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.906 = impl_witness_access constants.%OrderedWith.impl_witness.2de, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b8de4c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.621d39.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.c40960.1]
// CHECK:STDOUT:   %.loc20_5.3: %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.1 = specific_constant imports.%Core.Less.bb5, @T.as_type.as.OrderedWith.impl.804(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b8de4c.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.cd7b2d.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.b8de4c.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.621d39.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.58f79a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn [concrete = constants.%bound_method.c40960.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.539c = impl_witness_access constants.%OrderedWith.impl_witness.2de, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.308b69.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.b1f5e5.1]
// CHECK:STDOUT:   %.loc21_5.3: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.1 = specific_constant imports.%Core.GreaterOrEquivalent.6db, @T.as_type.as.OrderedWith.impl.804(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.52135f.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1b7ac6.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.308b69.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.4391cd.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn [concrete = constants.%bound_method.b1f5e5.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.33f]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.061 = impl_witness_access constants.%OrderedWith.impl_witness.2de, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.e3b7a4.1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.ee1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.7a8c59.1]
// CHECK:STDOUT:   %.loc22_5.3: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.1 = specific_constant imports.%Core.LessOrEquivalent.d62, @T.as_type.as.OrderedWith.impl.804(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b05ade.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.cf626d.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.e3b7a4.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.ee1) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.567cab.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn [concrete = constants.%bound_method.7a8c59.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.98e: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.15d: %T.as_type.as.ImplicitAs.impl.Convert.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.ada = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.9f8: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.2ef: %ImplicitAs.type.3c9 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.97b6f8.1: %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.97b6f8.2: %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.78e: <witness> = impl_witness imports.%Copy.impl_witness_table.842 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.78e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.74e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f48, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.03f: %ImplicitAs.type.3c9 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.74e) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.b61: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.627, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.457fcf.1: %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.457fcf.2: %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.c23: %AddAssignWith.type.9f8 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.844: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.c23) [concrete]
// CHECK:STDOUT:   %.df3: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.844, %AddAssignWith.facet.c23 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f5950e.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.457fcf.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f5950e.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.457fcf.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.d58: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c87dd1.1: %Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c87dd1.2: %Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.946: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.f8c, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bc06fc.1: %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bc06fc.2: %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.ce7: %SubAssignWith.type.d58 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.946) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.348: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.ce7) [concrete]
// CHECK:STDOUT:   %.f8b: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.348, %SubAssignWith.facet.ce7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.8e9580.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.bc06fc.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.8e9580.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.bc06fc.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.1a8: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.65a267.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.2c9ea0.1: %Cpp.long.as.MulAssignWith.impl.Op.type.65a267.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.65a267.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.2c9ea0.2: %Cpp.long.as.MulAssignWith.impl.Op.type.65a267.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.ad9: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.b66, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4b95a1.1: %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4b95a1.2: %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.2a7: %MulAssignWith.type.1a8 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.ad9) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.ff6: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.2a7) [concrete]
// CHECK:STDOUT:   %.a77: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.ff6, %MulAssignWith.facet.2a7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.e23e59.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4b95a1.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.e23e59.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4b95a1.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.14a: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.333e9d.1: %Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.333e9d.2: %Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.cc7: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.b64, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.dad0e5.1: %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.dad0e5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.0bf: %DivAssignWith.type.14a = facet_value %Cpp.long, (%DivAssignWith.impl_witness.cc7) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.4f6: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.0bf) [concrete]
// CHECK:STDOUT:   %.549: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.4f6, %DivAssignWith.facet.0bf [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.44bad8.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.dad0e5.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.44bad8.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.dad0e5.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.8a8: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.002543.1: %Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.002543.2: %Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.f24: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.616, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.92a81f.1: %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.92a81f.2: %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.400: %ModAssignWith.type.8a8 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.f24) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.511: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.400) [concrete]
// CHECK:STDOUT:   %.901: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.511, %ModAssignWith.facet.400 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.291265.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.92a81f.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.291265.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.92a81f.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.c11: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.fb3b66.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.fb3b66.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.ce4: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.288, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.9ac: %BitAndAssignWith.type.c11 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.ce4) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.ea4: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.9ac) [concrete]
// CHECK:STDOUT:   %.f6e: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.ea4, %BitAndAssignWith.facet.9ac [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.397f9a.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.397f9a.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.2f5: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.c04b02.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.c04b02.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.8ae: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.ab8, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.424: %BitOrAssignWith.type.2f5 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.8ae) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.24d: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.424) [concrete]
// CHECK:STDOUT:   %.8d9: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.24d, %BitOrAssignWith.facet.424 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.82af8c.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.82af8c.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.379: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.8644ea.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.8644ea.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.765: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.f38, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.e65: %BitXorAssignWith.type.379 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.765) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.6bc: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.e65) [concrete]
// CHECK:STDOUT:   %.e4e: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.6bc, %BitXorAssignWith.facet.e65 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.286f4e.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.286f4e.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.eec: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f10195.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f10195.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.ca9: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.6ad, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.aa0: %LeftShiftAssignWith.type.eec = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.ca9) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.c8d: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.aa0) [concrete]
// CHECK:STDOUT:   %.6ab: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.c8d, %LeftShiftAssignWith.facet.aa0 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.f099c2.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.f099c2.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.690: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.534d8b.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.534d8b.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.1b2: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.b67, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.dc2: %RightShiftAssignWith.type.690 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.1b2) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.7f2: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.dc2) [concrete]
// CHECK:STDOUT:   %.84d: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.7f2, %RightShiftAssignWith.facet.dc2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.e749ef.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.03f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.e749ef.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.03f) [concrete]
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
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a2d: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.98e) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.15d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f48 = impl_witness_table (%Core.import_ref.a2d), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.2c3: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.97b6f8.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.627 = impl_witness_table (%Core.import_ref.2c3), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.828: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.97b6f8.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.842 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.4c3: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c87dd1.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.f8c = impl_witness_table (%Core.import_ref.4c3), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe0: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.a1c5d0.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c87dd1.2)]
// CHECK:STDOUT:   %Core.import_ref.2c2: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.65a267.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.2c9ea0.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.b66 = impl_witness_table (%Core.import_ref.2c2), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.91a: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.65a267.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.2c9ea0.2)]
// CHECK:STDOUT:   %Core.import_ref.fca: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.333e9d.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.b64 = impl_witness_table (%Core.import_ref.fca), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.341: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.fb56d9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.333e9d.2)]
// CHECK:STDOUT:   %Core.import_ref.5d6: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.002543.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.616 = impl_witness_table (%Core.import_ref.5d6), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.894: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.3d7ddc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.002543.2)]
// CHECK:STDOUT:   %Core.import_ref.53b: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.fb3b66.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.288 = impl_witness_table (%Core.import_ref.53b), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.0a6: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.4fd52d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.fb3b66.2)]
// CHECK:STDOUT:   %Core.import_ref.e41: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.c04b02.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.ab8 = impl_witness_table (%Core.import_ref.e41), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.56f: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.035048.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.c04b02.2)]
// CHECK:STDOUT:   %Core.import_ref.36c: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.8644ea.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.f38 = impl_witness_table (%Core.import_ref.36c), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.310: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.01a7c8.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.8644ea.2)]
// CHECK:STDOUT:   %Core.import_ref.160: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f10195.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.6ad = impl_witness_table (%Core.import_ref.160), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.23c: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.837768.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f10195.2)]
// CHECK:STDOUT:   %Core.import_ref.365: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.534d8b.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.b67 = impl_witness_table (%Core.import_ref.365), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.14b: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.283894.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.534d8b.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.df3 = impl_witness_access constants.%AddAssignWith.impl_witness.b61, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.457fcf.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f5950e.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.1 = specific_constant imports.%Core.Op.828, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.457fcf.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.16674d.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.457fcf.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.f5950e.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.f8b = impl_witness_access constants.%SubAssignWith.impl_witness.946, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.bc06fc.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.8e9580.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.1 = specific_constant imports.%Core.Op.fe0, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.bc06fc.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.56cee2.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.bc06fc.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.8e9580.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.a77 = impl_witness_access constants.%MulAssignWith.impl_witness.ad9, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4b95a1.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.e23e59.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.1 = specific_constant imports.%Core.Op.91a, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4b95a1.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.e5ffc2.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4b95a1.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.e23e59.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.549 = impl_witness_access constants.%DivAssignWith.impl_witness.cc7, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.dad0e5.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.44bad8.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.1 = specific_constant imports.%Core.Op.341, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.dad0e5.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.fba539.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.dad0e5.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.44bad8.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.901 = impl_witness_access constants.%ModAssignWith.impl_witness.f24, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.92a81f.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.291265.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.1 = specific_constant imports.%Core.Op.894, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.92a81f.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.9ce69c.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.92a81f.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.291265.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.f6e = impl_witness_access constants.%BitAndAssignWith.impl_witness.ce4, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.397f9a.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.1 = specific_constant imports.%Core.Op.0a6, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.18b315.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.db86a3.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.397f9a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.8d9 = impl_witness_access constants.%BitOrAssignWith.impl_witness.8ae, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.82af8c.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.1 = specific_constant imports.%Core.Op.56f, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.448080.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9089cb.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.82af8c.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.e4e = impl_witness_access constants.%BitXorAssignWith.impl_witness.765, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.286f4e.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.1 = specific_constant imports.%Core.Op.310, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.833f63.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.a3666d.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.286f4e.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.6ab = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.ca9, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.f099c2.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.1 = specific_constant imports.%Core.Op.23c, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.c87f3e.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.addd0b.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.f099c2.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.84d = impl_witness_access constants.%RightShiftAssignWith.impl_witness.1b2, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.3c9 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.74e) [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.3c9 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.03f]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.e749ef.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.1 = specific_constant imports.%Core.Op.14b, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7835a4.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.05cc9d.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.03f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.e749ef.2]
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
// CHECK:STDOUT:   %Int.ec967b.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.9f8: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3758b9.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.ec967b.1)> [symbolic]
// CHECK:STDOUT:   %U.722: %ImplicitAs.type.3758b9.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.1: %Int.as.AddAssignWith.impl.Op.type.8adbfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.2: %Int.as.AddAssignWith.impl.Op.type.8adbfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.025: %ImplicitAs.type.270 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3f6: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.572, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.7773fb.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.da5bbf.1: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.7773fb.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.da5bbf.2: %Int.as.AddAssignWith.impl.Op.type.7773fb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.af8: %AddAssignWith.type.9f8 = facet_value %i64, (%AddAssignWith.impl_witness.3f6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d15: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.af8) [concrete]
// CHECK:STDOUT:   %.10d: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d15, %AddAssignWith.facet.af8 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.da5bbf.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.da5bbf.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47a, %ImplicitAs.facet.025 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.d58: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.dacc8a.1: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.bd5b18.1: %Int.as.SubAssignWith.impl.Op.type.dacc8a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.dacc8a.2: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.bd5b18.2: %Int.as.SubAssignWith.impl.Op.type.dacc8a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.ae7: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.27e, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.8b91e3.1: type = fn_type @Int.as.SubAssignWith.impl.Op.2, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.9e6ad1.1: %Int.as.SubAssignWith.impl.Op.type.8b91e3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.8b91e3.2: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.9e6ad1.2: %Int.as.SubAssignWith.impl.Op.type.8b91e3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.29f: %SubAssignWith.type.d58 = facet_value %i64, (%SubAssignWith.impl_witness.ae7) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.b90: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.29f) [concrete]
// CHECK:STDOUT:   %.1bd: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.b90, %SubAssignWith.facet.29f [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.dccb65.1: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.9e6ad1.2, @Int.as.SubAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.dccb65.2: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.9e6ad1.1, @Int.as.SubAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.1a8: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.2b981d.1: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.bc679d.1: %Int.as.MulAssignWith.impl.Op.type.2b981d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.2b981d.2: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.bc679d.2: %Int.as.MulAssignWith.impl.Op.type.2b981d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.f0c: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.96a, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.027d60.1: type = fn_type @Int.as.MulAssignWith.impl.Op.2, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.eacc45.1: %Int.as.MulAssignWith.impl.Op.type.027d60.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.027d60.2: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.eacc45.2: %Int.as.MulAssignWith.impl.Op.type.027d60.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.11f: %MulAssignWith.type.1a8 = facet_value %i64, (%MulAssignWith.impl_witness.f0c) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.8db: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.11f) [concrete]
// CHECK:STDOUT:   %.1d2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.8db, %MulAssignWith.facet.11f [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.4dcd23.1: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.eacc45.2, @Int.as.MulAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.4dcd23.2: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.eacc45.1, @Int.as.MulAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.14a: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.c1ae0a.1: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.f937e4.1: %Int.as.DivAssignWith.impl.Op.type.c1ae0a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.c1ae0a.2: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.f937e4.2: %Int.as.DivAssignWith.impl.Op.type.c1ae0a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.b55: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.4c8, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.f0c645.1: type = fn_type @Int.as.DivAssignWith.impl.Op.2, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.d79808.1: %Int.as.DivAssignWith.impl.Op.type.f0c645.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.f0c645.2: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.d79808.2: %Int.as.DivAssignWith.impl.Op.type.f0c645.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.41f: %DivAssignWith.type.14a = facet_value %i64, (%DivAssignWith.impl_witness.b55) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.e14: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.41f) [concrete]
// CHECK:STDOUT:   %.c31: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.e14, %DivAssignWith.facet.41f [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.5d2e5a.1: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.d79808.2, @Int.as.DivAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.5d2e5a.2: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.d79808.1, @Int.as.DivAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.8a8: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.b85781.1: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.2d4b01.1: %Int.as.ModAssignWith.impl.Op.type.b85781.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.b85781.2: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.2d4b01.2: %Int.as.ModAssignWith.impl.Op.type.b85781.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.bbc: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.5d5, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.ee8403.1: type = fn_type @Int.as.ModAssignWith.impl.Op.2, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.19c880.1: %Int.as.ModAssignWith.impl.Op.type.ee8403.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.ee8403.2: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.19c880.2: %Int.as.ModAssignWith.impl.Op.type.ee8403.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.35f: %ModAssignWith.type.8a8 = facet_value %i64, (%ModAssignWith.impl_witness.bbc) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.10c: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.35f) [concrete]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.10c, %ModAssignWith.facet.35f [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.0d9123.1: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.19c880.2, @Int.as.ModAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.0d9123.2: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.19c880.1, @Int.as.ModAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.c11: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.53ea52.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.444160.1: %Int.as.BitAndAssignWith.impl.Op.type.53ea52.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.53ea52.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.444160.2: %Int.as.BitAndAssignWith.impl.Op.type.53ea52.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.2d4: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.557, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.2, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.e2d922.1: %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.e2d922.2: %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.2f7: %BitAndAssignWith.type.c11 = facet_value %i64, (%BitAndAssignWith.impl_witness.2d4) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.00e: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.2f7) [concrete]
// CHECK:STDOUT:   %.48d: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.00e, %BitAndAssignWith.facet.2f7 [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.cc6f20.1: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.e2d922.2, @Int.as.BitAndAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.cc6f20.2: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.e2d922.1, @Int.as.BitAndAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.2f5: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.a12d86.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.b7a391.1: %Int.as.BitOrAssignWith.impl.Op.type.a12d86.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.a12d86.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.b7a391.2: %Int.as.BitOrAssignWith.impl.Op.type.a12d86.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.814: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.df2, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.73e218.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.2, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.c4d266.1: %Int.as.BitOrAssignWith.impl.Op.type.73e218.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.73e218.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.c4d266.2: %Int.as.BitOrAssignWith.impl.Op.type.73e218.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.a18: %BitOrAssignWith.type.2f5 = facet_value %i64, (%BitOrAssignWith.impl_witness.814) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.a1e: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.a18) [concrete]
// CHECK:STDOUT:   %.e20: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.a1e, %BitOrAssignWith.facet.a18 [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.75752b.1: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.c4d266.2, @Int.as.BitOrAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.75752b.2: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.c4d266.1, @Int.as.BitOrAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.379: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.de94c4.1: %Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.de94c4.2: %Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.94f: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.4b1, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.3c2326.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.2, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.33e181.1: %Int.as.BitXorAssignWith.impl.Op.type.3c2326.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.3c2326.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.33e181.2: %Int.as.BitXorAssignWith.impl.Op.type.3c2326.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.a9c: %BitXorAssignWith.type.379 = facet_value %i64, (%BitXorAssignWith.impl_witness.94f) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.662: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.a9c) [concrete]
// CHECK:STDOUT:   %.fec: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.662, %BitXorAssignWith.facet.a9c [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.3b2b73.1: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.33e181.2, @Int.as.BitXorAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.3b2b73.2: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.33e181.1, @Int.as.BitXorAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.eec: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.e8e815.1: %Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.e8e815.2: %Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.303: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.2b5, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.2, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.8856a6.1: %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.8856a6.2: %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.031: %LeftShiftAssignWith.type.eec = facet_value %i64, (%LeftShiftAssignWith.impl_witness.303) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.987: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.031) [concrete]
// CHECK:STDOUT:   %.368: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.987, %LeftShiftAssignWith.facet.031 [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4c0cdd.1: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.8856a6.2, @Int.as.LeftShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4c0cdd.2: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.8856a6.1, @Int.as.LeftShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.690: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.b96681.1: %Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.b96681.2: %Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.d4b: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.3b1, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.471d21.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.2, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.cef337.1: %Int.as.RightShiftAssignWith.impl.Op.type.471d21.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.471d21.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.cef337.2: %Int.as.RightShiftAssignWith.impl.Op.type.471d21.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.00f: %RightShiftAssignWith.type.690 = facet_value %i64, (%RightShiftAssignWith.impl_witness.d4b) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.0e9: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.00f) [concrete]
// CHECK:STDOUT:   %.a40: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.0e9, %RightShiftAssignWith.facet.00f [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.f8dcc3.1: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.cef337.2, @Int.as.RightShiftAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.f8dcc3.2: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.cef337.1, @Int.as.RightShiftAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.3c3: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.572 = impl_witness_table (%Core.import_ref.3c3), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b5f: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.825 [concrete]
// CHECK:STDOUT:   %Core.import_ref.19c: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.2 (%Int.as.SubAssignWith.impl.Op.type.dacc8a.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.2 (constants.%Int.as.SubAssignWith.impl.Op.bd5b18.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.27e = impl_witness_table (%Core.import_ref.19c), @Int.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.26f: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.1 (%Int.as.SubAssignWith.impl.Op.type.dacc8a.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.1 (constants.%Int.as.SubAssignWith.impl.Op.bd5b18.2)]
// CHECK:STDOUT:   %Core.import_ref.32f: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.2 (%Int.as.MulAssignWith.impl.Op.type.2b981d.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.2 (constants.%Int.as.MulAssignWith.impl.Op.bc679d.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.96a = impl_witness_table (%Core.import_ref.32f), @Int.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b9c: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.1 (%Int.as.MulAssignWith.impl.Op.type.2b981d.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.1 (constants.%Int.as.MulAssignWith.impl.Op.bc679d.2)]
// CHECK:STDOUT:   %Core.import_ref.115: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.2 (%Int.as.DivAssignWith.impl.Op.type.c1ae0a.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.2 (constants.%Int.as.DivAssignWith.impl.Op.f937e4.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.4c8 = impl_witness_table (%Core.import_ref.115), @Int.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.56a: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.1 (%Int.as.DivAssignWith.impl.Op.type.c1ae0a.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.1 (constants.%Int.as.DivAssignWith.impl.Op.f937e4.2)]
// CHECK:STDOUT:   %Core.import_ref.7b0: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.2 (%Int.as.ModAssignWith.impl.Op.type.b85781.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.2 (constants.%Int.as.ModAssignWith.impl.Op.2d4b01.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.5d5 = impl_witness_table (%Core.import_ref.7b0), @Int.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7b2: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.1 (%Int.as.ModAssignWith.impl.Op.type.b85781.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.1 (constants.%Int.as.ModAssignWith.impl.Op.2d4b01.2)]
// CHECK:STDOUT:   %Core.import_ref.e1e: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.2 (%Int.as.BitAndAssignWith.impl.Op.type.53ea52.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.2 (constants.%Int.as.BitAndAssignWith.impl.Op.444160.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.557 = impl_witness_table (%Core.import_ref.e1e), @Int.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.467: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.1 (%Int.as.BitAndAssignWith.impl.Op.type.53ea52.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.1 (constants.%Int.as.BitAndAssignWith.impl.Op.444160.2)]
// CHECK:STDOUT:   %Core.import_ref.d46: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.2 (%Int.as.BitOrAssignWith.impl.Op.type.a12d86.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.2 (constants.%Int.as.BitOrAssignWith.impl.Op.b7a391.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.df2 = impl_witness_table (%Core.import_ref.d46), @Int.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.79d: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.1 (%Int.as.BitOrAssignWith.impl.Op.type.a12d86.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.1 (constants.%Int.as.BitOrAssignWith.impl.Op.b7a391.2)]
// CHECK:STDOUT:   %Core.import_ref.230: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.2 (%Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.2 (constants.%Int.as.BitXorAssignWith.impl.Op.de94c4.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.4b1 = impl_witness_table (%Core.import_ref.230), @Int.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.08a: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.1 (%Int.as.BitXorAssignWith.impl.Op.type.b0d6d6.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.1 (constants.%Int.as.BitXorAssignWith.impl.Op.de94c4.2)]
// CHECK:STDOUT:   %Core.import_ref.8a4: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.2 (%Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.2 (constants.%Int.as.LeftShiftAssignWith.impl.Op.e8e815.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.2b5 = impl_witness_table (%Core.import_ref.8a4), @Int.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d08: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.1 (%Int.as.LeftShiftAssignWith.impl.Op.type.280ca1.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.1 (constants.%Int.as.LeftShiftAssignWith.impl.Op.e8e815.2)]
// CHECK:STDOUT:   %Core.import_ref.8fa4: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.2 (%Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.2 (constants.%Int.as.RightShiftAssignWith.impl.Op.b96681.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.3b1 = impl_witness_table (%Core.import_ref.8fa4), @Int.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.587: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.1 (%Int.as.RightShiftAssignWith.impl.Op.type.06b5e1.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.1 (constants.%Int.as.RightShiftAssignWith.impl.Op.b96681.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.10d = impl_witness_access constants.%AddAssignWith.impl_witness.3f6, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = specific_constant imports.%Core.Op.b5f, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %i64 = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.1bd = impl_witness_access constants.%SubAssignWith.impl_witness.ae7, element0 [concrete = constants.%Int.as.SubAssignWith.impl.Op.9e6ad1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Int.as.SubAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.dccb65.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %Int.as.SubAssignWith.impl.Op.type.8b91e3.1 = specific_constant imports.%Core.Op.26f, @Int.as.SubAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.SubAssignWith.impl.Op.9e6ad1.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Int.as.SubAssignWith.impl.Op.type.8b91e3.1 = name_ref Op, %.loc11_5 [concrete = constants.%Int.as.SubAssignWith.impl.Op.9e6ad1.1]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Int.as.SubAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.SubAssignWith.impl.Op.specific_fn.dccb65.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Int.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %i64 = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.1d2 = impl_witness_access constants.%MulAssignWith.impl_witness.f0c, element0 [concrete = constants.%Int.as.MulAssignWith.impl.Op.eacc45.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Int.as.MulAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.4dcd23.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %Int.as.MulAssignWith.impl.Op.type.027d60.1 = specific_constant imports.%Core.Op.b9c, @Int.as.MulAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.MulAssignWith.impl.Op.eacc45.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Int.as.MulAssignWith.impl.Op.type.027d60.1 = name_ref Op, %.loc12_5 [concrete = constants.%Int.as.MulAssignWith.impl.Op.eacc45.1]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Int.as.MulAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.MulAssignWith.impl.Op.specific_fn.4dcd23.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Int.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %i64 = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.c31 = impl_witness_access constants.%DivAssignWith.impl_witness.b55, element0 [concrete = constants.%Int.as.DivAssignWith.impl.Op.d79808.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Int.as.DivAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.5d2e5a.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %Int.as.DivAssignWith.impl.Op.type.f0c645.1 = specific_constant imports.%Core.Op.56a, @Int.as.DivAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.DivAssignWith.impl.Op.d79808.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Int.as.DivAssignWith.impl.Op.type.f0c645.1 = name_ref Op, %.loc13_5 [concrete = constants.%Int.as.DivAssignWith.impl.Op.d79808.1]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Int.as.DivAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.DivAssignWith.impl.Op.specific_fn.5d2e5a.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Int.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %i64 = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.f81 = impl_witness_access constants.%ModAssignWith.impl_witness.bbc, element0 [concrete = constants.%Int.as.ModAssignWith.impl.Op.19c880.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Int.as.ModAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.0d9123.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %Int.as.ModAssignWith.impl.Op.type.ee8403.1 = specific_constant imports.%Core.Op.7b2, @Int.as.ModAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.ModAssignWith.impl.Op.19c880.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Int.as.ModAssignWith.impl.Op.type.ee8403.1 = name_ref Op, %.loc14_5 [concrete = constants.%Int.as.ModAssignWith.impl.Op.19c880.1]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Int.as.ModAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.ModAssignWith.impl.Op.specific_fn.0d9123.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Int.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %i64 = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.48d = impl_witness_access constants.%BitAndAssignWith.impl_witness.2d4, element0 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.e2d922.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Int.as.BitAndAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.cc6f20.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5: %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.1 = specific_constant imports.%Core.Op.467, @Int.as.BitAndAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.e2d922.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Int.as.BitAndAssignWith.impl.Op.type.eb36f8.1 = name_ref Op, %.loc16_5 [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.e2d922.1]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Int.as.BitAndAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.cc6f20.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Int.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16: init %i64 = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %i64 = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.e20 = impl_witness_access constants.%BitOrAssignWith.impl_witness.814, element0 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.c4d266.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Int.as.BitOrAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.75752b.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5: %Int.as.BitOrAssignWith.impl.Op.type.73e218.1 = specific_constant imports.%Core.Op.79d, @Int.as.BitOrAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.c4d266.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Int.as.BitOrAssignWith.impl.Op.type.73e218.1 = name_ref Op, %.loc17_5 [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.c4d266.1]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Int.as.BitOrAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.75752b.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Int.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17: init %i64 = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %i64 = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.fec = impl_witness_access constants.%BitXorAssignWith.impl_witness.94f, element0 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.33e181.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Int.as.BitXorAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.3b2b73.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5: %Int.as.BitXorAssignWith.impl.Op.type.3c2326.1 = specific_constant imports.%Core.Op.08a, @Int.as.BitXorAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.33e181.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Int.as.BitXorAssignWith.impl.Op.type.3c2326.1 = name_ref Op, %.loc18_5 [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.33e181.1]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Int.as.BitXorAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.3b2b73.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Int.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18: init %i64 = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %i64 = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.368 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.303, element0 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.8856a6.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Int.as.LeftShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4c0cdd.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5: %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.1 = specific_constant imports.%Core.Op.d08, @Int.as.LeftShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.8856a6.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Int.as.LeftShiftAssignWith.impl.Op.type.56cf01.1 = name_ref Op, %.loc19_5 [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.8856a6.1]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Int.as.LeftShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.4c0cdd.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Int.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19: init %i64 = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %i64 = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %i64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.a40 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.d4b, element0 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.cef337.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Int.as.RightShiftAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.f8dcc3.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5: %Int.as.RightShiftAssignWith.impl.Op.type.471d21.1 = specific_constant imports.%Core.Op.587, @Int.as.RightShiftAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.cef337.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Int.as.RightShiftAssignWith.impl.Op.type.471d21.1 = name_ref Op, %.loc20_5 [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.cef337.1]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Int.as.RightShiftAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.f8dcc3.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Int.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.144: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.2ef: %ImplicitAs.type.3c9 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.97b6f8.1: %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.2ef) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.97b6f8.2: %Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.200: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.627, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.56ddbc.1: %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.56ddbc.2: %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.8ed: %AddAssignWith.type.144 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.200) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.fd2: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.8ed) [concrete]
// CHECK:STDOUT:   %.417: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.fd2, %AddAssignWith.facet.8ed [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9c08d1.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.56ddbc.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9c08d1.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.56ddbc.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.2c3: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.97b6f8.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.627 = impl_witness_table (%Core.import_ref.2c3), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.828: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.57bd5a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.97b6f8.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.417 = impl_witness_access constants.%AddAssignWith.impl_witness.200, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.56ddbc.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.157) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.3c9 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9c08d1.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.1 = specific_constant imports.%Core.Op.828, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.56ddbc.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.533c34.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.56ddbc.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.9c08d1.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %Int.ec967b.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.6d9: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6d9 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.6d9 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.270: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.122: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.791 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5f8: %ImplicitAs.type.270 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.122) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.de7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.5f8) [concrete]
// CHECK:STDOUT:   %.19a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.de7, %ImplicitAs.facet.5f8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.64f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.4f6: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long64 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.ada = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ee1: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.ee1) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet.ee1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.9f8: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3758b9.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.ec967b.1)> [symbolic]
// CHECK:STDOUT:   %U.722: %ImplicitAs.type.3758b9.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.1: %Int.as.AddAssignWith.impl.Op.type.8adbfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.2: %Int.as.AddAssignWith.impl.Op.type.8adbfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.025: %ImplicitAs.type.270 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.ddc) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3f6: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.572, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.7773fb.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.da5bbf.1: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.7773fb.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.da5bbf.2: %Int.as.AddAssignWith.impl.Op.type.7773fb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.af8: %AddAssignWith.type.9f8 = facet_value %i64, (%AddAssignWith.impl_witness.3f6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d15: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.af8) [concrete]
// CHECK:STDOUT:   %.10d: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d15, %AddAssignWith.facet.af8 [concrete]
// CHECK:STDOUT:   %self.param_patt.2fb: %pattern_type.6d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e26: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt.2fb [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.da5bbf.2, @Int.as.AddAssignWith.impl.Op.1(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.da5bbf.1, @Int.as.AddAssignWith.impl.Op.2(%int_64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.025) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47a, %ImplicitAs.facet.025 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.3c3: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.572 = impl_witness_table (%Core.import_ref.3c3), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b5f: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.06c923.2), @Cpp.long.as.ImplicitAs.impl.825 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6d9 = ref_param_pattern [concrete = constants.%self.param_patt.2fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e26]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.19a = impl_witness_access constants.%ImplicitAs.impl_witness.122, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.64f]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d8f]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i64 = call %bound_method.loc8_3.2(%int_1.loc8) [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   %.loc8: init %i64 = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.4f6]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %a: ref %i64 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6d9 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6d9 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.10d = impl_witness_access constants.%AddAssignWith.impl_witness.3f6, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Int.as.AddAssignWith.impl.Op.1(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = specific_constant imports.%Core.Op.b5f, @Int.as.AddAssignWith.impl(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.7773fb.1 = name_ref Op, %.loc10_5 [concrete = constants.%Int.as.AddAssignWith.impl.Op.da5bbf.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.025) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.7b7c3d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.dca = impl_witness_access constants.%ImplicitAs.impl_witness.ddc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.type.3c9: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.157: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.68f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.157) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.908: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.7d4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.908, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.33f: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.5c2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.915: type = fn_type_with_self_type %Inc.WithSelf.Op.type.5c2, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fe4: %pattern_type.ada = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0d8: %pattern_type.ada = wrapper_binding_pattern self, %self.param_patt.fe4 [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.1cf: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.203: type = fn_type_with_self_type %Dec.WithSelf.Op.type.1cf, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.68f = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.9b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long.as.Dec.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7d4 = impl_witness_access constants.%ImplicitAs.impl_witness.157, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.915 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.203 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
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
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_long_long.patt: %pattern_type.878 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.615: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.7db(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.edd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bf2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bae = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0ef: %ImplicitAs.type.615 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a86) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b73: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.0ef) [concrete]
// CHECK:STDOUT:   %.072: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b73, %ImplicitAs.facet.0ef [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4bb: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d00: %u64 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.62c: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1bb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.444 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8d1: %ImplicitAs.type.62c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1bb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.23a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.8d1) [concrete]
// CHECK:STDOUT:   %.90c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.23a, %ImplicitAs.facet.8d1 [concrete]
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
// CHECK:STDOUT:   %carbon_u64.patt: %pattern_type.878 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   %carbon_u64_result.patt: %pattern_type.97e = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.1b7: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.4c4: <witness> = impl_witness imports.%As.impl_witness_table.725 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.1b7 = facet_value Core.IntLiteral, (%As.impl_witness.4c4) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f25: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.a36: type = fn_type_with_self_type %As.WithSelf.Convert.type.f25, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.db9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d28 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4a1: %ImplicitAs.type.5be = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.db9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.10e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.4a1) [concrete]
// CHECK:STDOUT:   %.bfb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.10e, %ImplicitAs.facet.4a1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e8, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.602: @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.edd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bf2 = impl_witness_table (%Core.import_ref.602), @Core.IntLiteral.as.ImplicitAs.impl.7db [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.444 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.716 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.725 = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.ac7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d28 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long.as.ImplicitAs.impl.b87 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0f1 = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.072 = impl_witness_access constants.%ImplicitAs.impl_witness.a86, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d]
// CHECK:STDOUT:   %bound_method.loc15_56.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e2]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_56.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.4bb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %u64 = call %bound_method.loc15_56.2(%int_1.loc15) [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc15_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc15_56.2: %u64 = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = wrapper_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.878 = value_binding_pattern cpp_unsigned_long_long [concrete = constants.%cpp_unsigned_long_long.patt]
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
// CHECK:STDOUT:   %impl.elem0.loc18: %.90c = impl_witness_access constants.%ImplicitAs.impl_witness.1bb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
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
// CHECK:STDOUT:   %impl.elem0.loc21: %.072 = impl_witness_access constants.%ImplicitAs.impl_witness.a86, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6d]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e2]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.4bb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.d00]
// CHECK:STDOUT:   %u64.loc21: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_u64: %u64 = wrapper_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.878 = value_binding_pattern carbon_u64 [concrete = constants.%carbon_u64.patt]
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
// CHECK:STDOUT:   %a.var: ref %Array.984 = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.2d9 = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.2d9 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.2d9 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.2d9 to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc24_59.5: init %Array.984 to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %Array.984 = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %Array [concrete = constants.%Array.984] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.a36 = impl_witness_access constants.%As.impl_witness.4c4, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.9e8]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.bfb = impl_witness_access constants.%ImplicitAs.impl_witness.db9, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %.loc22_83.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%.loc22_83.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %.loc19_93.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%.loc19_93.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %.loc16_107.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
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
// CHECK:STDOUT:   %ImplicitAs.type.62c: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1bb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.444 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.62c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1bb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.23a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.90c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.23a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.9e8: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.172 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.172 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2bb: <witness> = impl_witness imports.%Copy.impl_witness_table.2f0 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.2bb) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.0c9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cdb: type = fn_type_with_self_type %Copy.WithSelf.Op.type.0c9, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.878: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %self.param_patt.4cb: %pattern_type.878 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.198: %pattern_type.878 = wrapper_binding_pattern self, %self.param_patt.4cb [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.444 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.716 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2f0 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.90c = impl_witness_access constants.%ImplicitAs.impl_witness.1bb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.cdb = impl_witness_access constants.%Copy.impl_witness.2bb, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
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
