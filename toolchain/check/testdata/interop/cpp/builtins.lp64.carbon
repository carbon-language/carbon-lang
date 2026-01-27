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
  let carbon_long: i64 = cpp_long;

  var a: array(f32, 1 as Cpp.long) = (1.0,);
  //@dump-sem-ir-end
}

// --- unsigned_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long: Cpp.unsigned_long = 1;
  let carbon_unsigned_long: u64 = cpp_unsigned_long;

  var a: array(f32, 1 as Cpp.unsigned_long) = (1.0,);
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
  let cpp_long_long_result: Cpp.LongLongResult = Cpp.PassLongLong(cpp_long_long);

  let cpp_compat_long_long: Core.CppCompat.LongLong64 = cpp_long_long;
  let cpp_compat_long_long_result: Cpp.LongLongResult = Cpp.PassLongLong(cpp_compat_long_long);

  let carbon_i64: i64 = cpp_long_long;
  let carbon_i64_result: Cpp.LongResult = Cpp.PassLongLong(carbon_i64);

  var a: array(f32, 1 as Cpp.long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- copy_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  var b: Cpp.long_long = a;
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
  //@dump-sem-ir-begin
  let a: Cpp.long_long = 1;
  a == (1 as i64);
  a != (1 as i64);
  a > (1 as i64);
  a < (1 as i64);
  a >= (1 as i64);
  a <= (1 as i64);

  a == 1;
  a != 1;
  a > 1;
  a < 1;
  a >= 1;
  a <= 1;

  let b: i64 = 1;
  a == b;
  a != b;
  a > b;
  a < b;
  a >= b;
  a <= b;
  //@dump-sem-ir-end
}

// --- comparisons_heterogeneous_long_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongRightSide() {
  //@dump-sem-ir-begin
  let a: Cpp.long_long = 1;
  (1 as i64) == a;
  (1 as i64) != a;
  (1 as i64) > a;
  (1 as i64) < a;
  (1 as i64) >= a;
  (1 as i64) <= a;

  1 == a;
  1 != a;
  1 > a;
  1 < a;
  1 >= a;
  1 <= a;

  let b: i64 = 1;
  b == a;
  b != a;
  b > a;
  b < a;
  b >= a;
  b <= a;
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

// --- fail_todo_comparisons_heterogeneous_long_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongAndI128() {
  let a: Cpp.long_long = 1;
  let b: i128 = 1;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i128.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(i128)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a == b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a == b;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i128.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(Cpp.long_long)` in type `i128` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   b == a;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  b == a;
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
  let cpp_unsigned_long_long_result: Cpp.ULongLongResult = Cpp.PassULongLong(cpp_unsigned_long_long);

  let cpp_compat_ulong_long: Core.CppCompat.ULongLong64 = cpp_unsigned_long_long;
  let cpp_compat_ulong_long_result: Cpp.ULongLongResult = Cpp.PassULongLong(cpp_compat_ulong_long);

  let carbon_u64: u64 = cpp_unsigned_long_long;
  let carbon_u64_result: Cpp.ULongResult = Cpp.PassULongLong(carbon_u64);

  var a: array(f32, 1 as Cpp.unsigned_long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- copy_unsigned_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyUnsignedLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.unsigned_long_long = 1;
  var b: Cpp.unsigned_long_long = a;
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.288: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %.aba: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.ccf: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.535: %Int.as.ImplicitAs.impl.Convert.type.ccf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.234: %ImplicitAs.type.139 = facet_value %i64, (%ImplicitAs.impl_witness.e86) [concrete]
// CHECK:STDOUT:   %.3aa: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.234 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.535 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.535, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d1e: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = @F.%i64.1
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
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
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
// CHECK:STDOUT:   %.loc9: type = splice_block %i64.loc9 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc11_43.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_43: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_43.1: <bound method> = bound_method %float, %impl.elem0.loc11_43 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_43: <specific function> = specific_function %impl.elem0.loc11_43, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_43.2: <bound method> = bound_method %float, %specific_fn.loc11_43 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_43.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_43.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_43.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_43.4: init %f32.97e to %.loc11_43.3 = initialize_from %.loc11_43.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_43.5: init %array_type to %a.var = array_init (%.loc11_43.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_43.1, %.loc11_43.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_34: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_23.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:     %bound_method.loc11_23.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.1: <specific function> = specific_function %impl.elem0.loc11_23.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_23.1 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_23.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_23.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_23.2: %i64 = converted %int_1.loc11, %.loc11_23.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc11_23.2: %.3aa = impl_witness_access constants.%ImplicitAs.impl_witness.e86, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.535]
// CHECK:STDOUT:     %bound_method.loc11_23.3: <bound method> = bound_method %.loc11_23.2, %impl.elem0.loc11_23.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.2: <specific function> = specific_function %impl.elem0.loc11_23.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.4: <bound method> = bound_method %.loc11_23.2, %specific_fn.loc11_23.2 [concrete = constants.%bound_method.d1e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_23.4(%.loc11_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.4: Core.IntLiteral = converted %.loc11_23.2, %.loc11_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %array_type) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.c65: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.7f8: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.0ea: %UInt.as.ImplicitAs.impl.Convert.type.7f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.042: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfe: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bb3) [concrete]
// CHECK:STDOUT:   %.da7: type = fn_type_with_self_type %ImplicitAs.Convert.type.c65, %ImplicitAs.facet.dfe [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.042 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.042, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.f20: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.979: %Core.IntLiteral.as.As.impl.Convert.type.2b1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.f54: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0f2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.715: %Core.IntLiteral.as.As.impl.Convert.type.0f2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.f20 = facet_value Core.IntLiteral, (%As.impl_witness.f54) [concrete]
// CHECK:STDOUT:   %.11a: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.715 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.715, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.9cf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2cc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.493, @UInt.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.812: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.309: %UInt.as.ImplicitAs.impl.Convert.type.812 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.02d: %ImplicitAs.type.139 = facet_value %u64, (%ImplicitAs.impl_witness.2cc) [concrete]
// CHECK:STDOUT:   %.a1c: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.02d [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.309 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.309, @UInt.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.f18: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = @F.%u64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.483: @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.7f8) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.0ea)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.493 = impl_witness_table (%Core.import_ref.483), @UInt.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.da7 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
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
// CHECK:STDOUT:   %.loc9: type = splice_block %u64.loc9 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc9: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc11_52.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_52: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_52.1: <bound method> = bound_method %float, %impl.elem0.loc11_52 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_52: <specific function> = specific_function %impl.elem0.loc11_52, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_52.2: <bound method> = bound_method %float, %specific_fn.loc11_52 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_52.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_52.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_52.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_52.4: init %f32.97e to %.loc11_52.3 = initialize_from %.loc11_52.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_52.5: init %array_type to %a.var = array_init (%.loc11_52.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_52.1, %.loc11_52.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_43: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_23.1: %.11a = impl_witness_access constants.%As.impl_witness.f54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.715]
// CHECK:STDOUT:     %bound_method.loc11_23.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.1: <specific function> = specific_function %impl.elem0.loc11_23.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_23.1 [concrete = constants.%bound_method.9cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_23.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_23.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_23.2: %u64 = converted %int_1.loc11, %.loc11_23.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc11_23.2: %.a1c = impl_witness_access constants.%ImplicitAs.impl_witness.2cc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.309]
// CHECK:STDOUT:     %bound_method.loc11_23.3: <bound method> = bound_method %.loc11_23.2, %impl.elem0.loc11_23.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.2: <specific function> = specific_function %impl.elem0.loc11_23.2, @UInt.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.4: <bound method> = bound_method %.loc11_23.2, %specific_fn.loc11_23.2 [concrete = constants.%bound_method.f18]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_23.4(%.loc11_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.4: Core.IntLiteral = converted %.loc11_23.2, %.loc11_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %array_type) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6e4: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.257: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.a95: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.aa8bde.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.9998b6.1: %PassLongLong__carbon_thunk.type.aa8bde.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a63: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.563 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.660: %ImplicitAs.type.2ad = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.a63) [concrete]
// CHECK:STDOUT:   %.912: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.660 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type.194: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.c03: %Cpp.long_long.as.ImplicitAs.impl.Convert.type.194 = struct_value () [concrete]
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
// CHECK:STDOUT:   %As.Convert.type.92c: type = fn_type @As.Convert, @As(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %As.impl_witness.158: <witness> = impl_witness imports.%As.impl_witness_table.eb3 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b3e = facet_value Core.IntLiteral, (%As.impl_witness.158) [concrete]
// CHECK:STDOUT:   %.ae2: type = fn_type_with_self_type %As.Convert.type.92c, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.737: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.64f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.084: %ImplicitAs.type.139 = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.737) [concrete]
// CHECK:STDOUT:   %.7c8: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.084 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type.506: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.ad2: %Cpp.long_long.as.ImplicitAs.impl.Convert.type.506 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.092, %Cpp.long_long.as.ImplicitAs.impl.Convert.ad2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc24 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.139b1c.1: %PassLongLong__carbon_thunk.type.aa8bde.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.9998b6.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.91c3: %Cpp.long_long.as.ImplicitAs.impl.Convert.type.194 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.c03]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.563 = impl_witness_table (%Core.import_ref.91c3), @Cpp.long_long.as.ImplicitAs.impl.034 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74a: %Cpp.long_long.as.ImplicitAs.impl.Convert.type.506 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.ad2]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc15_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc15_38.2: %Cpp.long_long = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = value_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.257 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16_80.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.a95 = addr_of %.loc16_80.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_80.2: init %LongLongResult to %.loc16_80.1 = in_place_init %PassLongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_32: type = splice_block %LongLongResult.ref.loc16 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc16: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_80.3: ref %LongLongResult = temporary %.loc16_80.1, %.loc16_80.2
// CHECK:STDOUT:   %.loc16_80.4: %LongLongResult = acquire_value %.loc16_80.3
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = value_binding cpp_long_long_result, %.loc16_80.4
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
// CHECK:STDOUT:   %Cpp.ref.loc19_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc19_94.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.a95 = addr_of %.loc19_94.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.1(%cpp_compat_long_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_94.2: init %LongLongResult to %.loc19_94.1 = in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_39: type = splice_block %LongLongResult.ref.loc19 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc19: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_94.3: ref %LongLongResult = temporary %.loc19_94.1, %.loc19_94.2
// CHECK:STDOUT:   %.loc19_94.4: %LongLongResult = acquire_value %.loc19_94.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = value_binding cpp_compat_long_long_result, %.loc19_94.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc21_19: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.c03]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %cpp_long_long.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc21: init %i64 = call %bound_method.loc21(%cpp_long_long.ref.loc21)
// CHECK:STDOUT:   %.loc21_25.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc21
// CHECK:STDOUT:   %.loc21_25.2: %i64 = converted %cpp_long_long.ref.loc21, %.loc21_25.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.cd9 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_70.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.305 = addr_of %.loc22_70.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.139b1c.2(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_70.2: init %LongResult to %.loc22_70.1 = in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_29: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_70.3: ref %LongResult = temporary %.loc22_70.1, %.loc22_70.2
// CHECK:STDOUT:   %.loc22_70.4: %LongResult = acquire_value %.loc22_70.3
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = value_binding carbon_i64_result, %.loc22_70.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc24_48.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_48: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc24_48.1: <bound method> = bound_method %float, %impl.elem0.loc24_48 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_48, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_48.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_48.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_48.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_48.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_48.4: init %f32.97e to %.loc24_48.3 = initialize_from %.loc24_48.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_48.5: init %array_type to %a.var = array_init (%.loc24_48.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_48.1, %.loc24_48.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_39: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_23.1: %.ae2 = impl_witness_access constants.%As.impl_witness.158, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_23.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_23.1(%int_1.loc24) [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_23.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %.loc24_23.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_23.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:     %impl.elem0.loc24_23.2: %.7c8 = impl_witness_access constants.%ImplicitAs.impl_witness.737, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.ad2]
// CHECK:STDOUT:     %bound_method.loc24_23.2: <bound method> = bound_method %.loc24_23.2, %impl.elem0.loc24_23.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc24: init Core.IntLiteral = call %bound_method.loc24_23.2(%.loc24_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.4: Core.IntLiteral = converted %.loc24_23.2, %.loc24_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_70.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call %LongResult.cpp_destructor.bound(%.loc22_70.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_94.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc19(%.loc19_94.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_80.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc16(%.loc16_80.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc24(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc22(%self.param: %LongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc19(%self.param: %LongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6e4: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6d4: <witness> = impl_witness imports.%Copy.impl_witness_table.804 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.6d4) [concrete]
// CHECK:STDOUT:   %.26f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %.loc9_26: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.26f = impl_witness_access constants.%Copy.impl_witness.6d4, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_26, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc9(%.loc9_26)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %b.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long_long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6e4: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.6d5: type = fn_type @EqWith.Equal, @EqWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.c18: type = fn_type @EqWith.NotEqual, @EqWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.625: <witness> = impl_witness imports.%EqWith.impl_witness_table.636 [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.4be = facet_value %Cpp.long_long, (%EqWith.impl_witness.625) [concrete]
// CHECK:STDOUT:   %.a50: type = fn_type_with_self_type %EqWith.Equal.type.6d5, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.7de: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.49b: %Cpp.long_long.as.EqWith.impl.Equal.type.7de = struct_value () [concrete]
// CHECK:STDOUT:   %.c9a: type = fn_type_with_self_type %EqWith.NotEqual.type.c18, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.4e3: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.608: %Cpp.long_long.as.EqWith.impl.NotEqual.type.4e3 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.bc1: type = fn_type @OrderedWith.Less, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.1cc: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.921: type = fn_type @OrderedWith.Greater, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.fcf: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.dab: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.366 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.65e = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.dab) [concrete]
// CHECK:STDOUT:   %.dbc: type = fn_type_with_self_type %OrderedWith.Greater.type.921, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.445: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.d29: %Cpp.long_long.as.OrderedWith.impl.Greater.type.445 = struct_value () [concrete]
// CHECK:STDOUT:   %.be1: type = fn_type_with_self_type %OrderedWith.Less.type.bc1, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a02: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.304: %Cpp.long_long.as.OrderedWith.impl.Less.type.a02 = struct_value () [concrete]
// CHECK:STDOUT:   %.03c: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.fcf, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.cd4: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.cd4 = struct_value () [concrete]
// CHECK:STDOUT:   %.c0e: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.1cc, %OrderedWith.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.a50 = impl_witness_access constants.%EqWith.impl_witness.625, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.49b]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.c9a = impl_witness_access constants.%EqWith.impl_witness.625, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.608]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.dbc = impl_witness_access constants.%OrderedWith.impl_witness.dab, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.d29]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.be1 = impl_witness_access constants.%OrderedWith.impl_witness.dab, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.304]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.03c = impl_witness_access constants.%OrderedWith.impl_witness.dab, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bcc]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.c0e = impl_witness_access constants.%OrderedWith.impl_witness.dab, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.de3]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6e4: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %.aba: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d68: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.59b: type = fn_type @EqWith.Equal, @EqWith(%i64) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.69a: type = fn_type @EqWith.NotEqual, @EqWith(%i64) [concrete]
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
// CHECK:STDOUT:   %EqWith.NotEqual.type.b22: type = fn_type @EqWith.NotEqual, @EqWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.f75: type = fn_type @EqWith.Equal, @EqWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %.6ba: type = fn_type_with_self_type %EqWith.Equal.type.59b, %EqWith.facet.d0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.610927.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.b29: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet.c59 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.41a, %i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.610927.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.838: type = fn_type_with_self_type %EqWith.NotEqual.type.69a, %EqWith.facet.d0f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.e34: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.f12: type = fn_type @OrderedWith.Less, @OrderedWith(%i64) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.c09: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%i64) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.3c7: type = fn_type @OrderedWith.Greater, @OrderedWith(%i64) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.d03: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%i64) [concrete]
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
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.e24: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.30c: type = fn_type @OrderedWith.Greater, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.776: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.20d: type = fn_type @OrderedWith.Less, @OrderedWith(Core.IntLiteral) [concrete]
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
// CHECK:STDOUT:   %.1fe: type = fn_type_with_self_type %OrderedWith.Greater.type.3c7, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.136328.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.136328.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.0e7: type = fn_type_with_self_type %OrderedWith.Less.type.f12, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.39610d.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.39610d.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.30d: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.d03, %OrderedWith.facet.149 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c59) [concrete]
// CHECK:STDOUT:   %.90d: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.c09, %OrderedWith.facet.149 [concrete]
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
// CHECK:STDOUT:   %.5a8: type = fn_type_with_self_type %EqWith.Equal.type.f75, %EqWith.facet.8cf [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.3190b1.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.3190b1.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.f04: type = fn_type_with_self_type %EqWith.NotEqual.type.b22, %EqWith.facet.8cf [concrete]
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
// CHECK:STDOUT:   %.490: type = fn_type_with_self_type %OrderedWith.Greater.type.30c, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.47c: type = fn_type_with_self_type %OrderedWith.Less.type.20d, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.f68: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.e24, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %.c1a: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.776, %OrderedWith.facet.8a0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.288: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.142: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.7b9c94.1)]
// CHECK:STDOUT:   %Core.import_ref.71f: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.902 = impl_witness_table (%Core.import_ref.142, %Core.import_ref.71f), @Cpp.long_long.as.EqWith.impl.1bc [concrete]
// CHECK:STDOUT:   %Core.NotEqual.dea: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.a5582c.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.e250a6.2)]
// CHECK:STDOUT:   %Core.Equal.357: @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.ec19bf.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.1bc.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.7b9c94.2)]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.4f4b: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b19 = impl_witness_table (%Core.import_ref.4f4b), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.81d: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.c164d2.1)]
// CHECK:STDOUT:   %Core.import_ref.21c: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.1)]
// CHECK:STDOUT:   %Core.import_ref.298: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.1)]
// CHECK:STDOUT:   %Core.import_ref.e3d: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.a1b = impl_witness_table (%Core.import_ref.81d, %Core.import_ref.21c, %Core.import_ref.298, %Core.import_ref.e3d), @Cpp.long_long.as.OrderedWith.impl.ef6 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.489: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.89fdb1.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.300fb6.2)]
// CHECK:STDOUT:   %Core.Greater.a10: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9fb493.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a3ca0e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.aaf: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.3ce459.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.8dc0bf.2)]
// CHECK:STDOUT:   %Core.Less.511: @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.d531c6.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.ef6.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.c164d2.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc8_26.2
// CHECK:STDOUT:   %a.ref.loc9: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc9_11.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc9_11.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_11.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9_11: <specific function> = specific_function %impl.elem0.loc9_11.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_11.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_11 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_11.2(%int_1.loc9) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_11.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc9 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_11.2: %i64 = converted %int_1.loc9, %.loc9_11.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.6ba = impl_witness_access constants.%EqWith.impl_witness.756, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc9_5: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref.loc9, %specific_fn.loc9_5
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = specific_constant imports.%Core.Equal.357, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Equal.ref.loc9: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = name_ref Equal, %.loc9_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc9: <bound method> = bound_method %a.ref.loc9, %Equal.ref.loc9
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %.loc9_11.2, %impl.elem0.loc9_5.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long_long = call %bound_method.loc9_5.3(%.loc9_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_5.5: %Cpp.long_long = converted %.loc9_11.2, %.loc9_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc9: <specific function> = specific_function %Equal.ref.loc9, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref.loc9, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc9
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_11.3: <bound method> = bound_method %.loc9_11.2, %impl.elem0.loc9_11.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc9_11: init %Cpp.long_long = call %bound_method.loc9_11.3(%.loc9_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_11.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc9_11 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc9_11.4: %Cpp.long_long = converted %.loc9_11.2, %.loc9_11.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc9: init bool = call %bound_method.loc9_5.4(%a.ref.loc9, %.loc9_11.4)
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc10: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc10: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_11.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10_11: <specific function> = specific_function %impl.elem0.loc10_11.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_11 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_11.2(%int_1.loc10) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc10_11.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc10 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc10_11.2: %i64 = converted %int_1.loc10, %.loc10_11.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem1.loc10: %.838 = impl_witness_access constants.%EqWith.impl_witness.756, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem1.loc10
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc10_5: <specific function> = specific_function %impl.elem1.loc10, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10_5
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = specific_constant imports.%Core.NotEqual.dea, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %NotEqual.ref.loc10: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = name_ref NotEqual, %.loc10_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc10: <bound method> = bound_method %a.ref.loc10, %NotEqual.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %.loc10_11.2, %impl.elem0.loc10_5 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long_long = call %bound_method.loc10_5.3(%.loc10_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc10_5.5: %Cpp.long_long = converted %.loc10_11.2, %.loc10_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc10: <specific function> = specific_function %NotEqual.ref.loc10, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.2, %impl.elem0.loc10_11.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10_11: init %Cpp.long_long = call %bound_method.loc10_11.3(%.loc10_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc10_11.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10_11 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc10_11.4: %Cpp.long_long = converted %.loc10_11.2, %.loc10_11.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_11.4)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc11: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc11: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc11_10.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_10.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_10: <specific function> = specific_function %impl.elem0.loc11_10.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_10 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_10.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_10.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_10.2: %i64 = converted %int_1.loc11, %.loc11_10.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem2.loc11: %.1fe = impl_witness_access constants.%OrderedWith.impl_witness.940, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem2.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc11_5: <specific function> = specific_function %impl.elem2.loc11, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11_5
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = specific_constant imports.%Core.Greater.a10, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Greater.ref.loc11: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = name_ref Greater, %.loc11_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc11: <bound method> = bound_method %a.ref.loc11, %Greater.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %.loc11_10.2, %impl.elem0.loc11_5 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long_long = call %bound_method.loc11_5.3(%.loc11_10.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc11_5.5: %Cpp.long_long = converted %.loc11_10.2, %.loc11_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc11: <specific function> = specific_function %Greater.ref.loc11, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_10.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_10.3: <bound method> = bound_method %.loc11_10.2, %impl.elem0.loc11_10.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11_10: init %Cpp.long_long = call %bound_method.loc11_10.3(%.loc11_10.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc11_10.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11_10 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc11_10.4: %Cpp.long_long = converted %.loc11_10.2, %.loc11_10.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_10.4)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc12: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc12: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc12_10.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc12_10.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_10.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc12_10: <specific function> = specific_function %impl.elem0.loc12_10.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_10.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_10 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_10.2(%int_1.loc12) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc12_10.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc12 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc12_10.2: %i64 = converted %int_1.loc12, %.loc12_10.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.0e7 = impl_witness_access constants.%OrderedWith.impl_witness.940, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc12_5: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12_5
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = specific_constant imports.%Core.Less.511, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Less.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = name_ref Less, %.loc12_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Less.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %.loc12_10.2, %impl.elem0.loc12_5.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long_long = call %bound_method.loc12_5.3(%.loc12_10.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc12_5.5: %Cpp.long_long = converted %.loc12_10.2, %.loc12_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc12: <specific function> = specific_function %Less.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_10.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_10.3: <bound method> = bound_method %.loc12_10.2, %impl.elem0.loc12_10.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12_10: init %Cpp.long_long = call %bound_method.loc12_10.3(%.loc12_10.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc12_10.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12_10 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc12_10.4: %Cpp.long_long = converted %.loc12_10.2, %.loc12_10.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_10.4)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc13: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc13: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc13_11.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc13_11.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_11.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13_11: <specific function> = specific_function %impl.elem0.loc13_11.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_11.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13_11 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_11.2(%int_1.loc13) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc13_11.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc13 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc13_11.2: %i64 = converted %int_1.loc13, %.loc13_11.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem3.loc13: %.30d = impl_witness_access constants.%OrderedWith.impl_witness.940, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem3.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc13_5: <specific function> = specific_function %impl.elem3.loc13, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13_5
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = specific_constant imports.%Core.GreaterOrEquivalent.489, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = name_ref GreaterOrEquivalent, %.loc13_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc13: <bound method> = bound_method %a.ref.loc13, %GreaterOrEquivalent.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %.loc13_11.2, %impl.elem0.loc13_5 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long_long = call %bound_method.loc13_5.3(%.loc13_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc13_5.5: %Cpp.long_long = converted %.loc13_11.2, %.loc13_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc13: <specific function> = specific_function %GreaterOrEquivalent.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_11.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_11.3: <bound method> = bound_method %.loc13_11.2, %impl.elem0.loc13_11.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13_11: init %Cpp.long_long = call %bound_method.loc13_11.3(%.loc13_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc13_11.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13_11 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc13_11.4: %Cpp.long_long = converted %.loc13_11.2, %.loc13_11.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_11.4)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc14: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc14: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc14_11.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_11.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc14_11: <specific function> = specific_function %impl.elem0.loc14_11.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_11 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_11.2(%int_1.loc14) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc14_11.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc14_11.2: %i64 = converted %int_1.loc14, %.loc14_11.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %impl.elem1.loc14: %.90d = impl_witness_access constants.%OrderedWith.impl_witness.940, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc14_5: <specific function> = specific_function %impl.elem1.loc14, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14_5
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = specific_constant imports.%Core.LessOrEquivalent.aaf, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = name_ref LessOrEquivalent, %.loc14_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %LessOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14_5 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long_long = call %bound_method.loc14_5.3(%.loc14_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc14_5.5: %Cpp.long_long = converted %.loc14_11.2, %.loc14_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc14: <specific function> = specific_function %LessOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_11.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_11.3: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14_11.2 [concrete = constants.%i64.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14_11: init %Cpp.long_long = call %bound_method.loc14_11.3(%.loc14_11.2) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc14_11.3: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14_11 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc14_11.4: %Cpp.long_long = converted %.loc14_11.2, %.loc14_11.3 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_11.4)
// CHECK:STDOUT:   %a.ref.loc16: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.5a8 = impl_witness_access constants.%EqWith.impl_witness.5f3, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc16_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc16_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc16_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1 = specific_constant imports.%Core.Equal.357, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.1]
// CHECK:STDOUT:   %Equal.ref.loc16: %Cpp.long_long.as.EqWith.impl.Equal.type.65adc5.1 = name_ref Equal, %.loc16_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.3190b1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc16: <bound method> = bound_method %a.ref.loc16, %Equal.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long_long = call %bound_method.loc16_5.3(%int_1.loc16) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_5.5: %Cpp.long_long = converted %int_1.loc16, %.loc16_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc16: <specific function> = specific_function %Equal.ref.loc16, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.72c761.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long_long = call %bound_method.loc16_8(%int_1.loc16) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %int_1.loc16, %.loc16_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc16: init bool = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc17: %.f04 = impl_witness_access constants.%EqWith.impl_witness.5f3, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem1.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem1.loc17, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1 = specific_constant imports.%Core.NotEqual.dea, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc17: %Cpp.long_long.as.EqWith.impl.NotEqual.type.84cb6f.1 = name_ref NotEqual, %.loc17_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.7e6eb6.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc17: <bound method> = bound_method %a.ref.loc17, %NotEqual.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long_long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_5.5: %Cpp.long_long = converted %int_1.loc17, %.loc17_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc17: <specific function> = specific_function %NotEqual.ref.loc17, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.ace954.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc18: %.490 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem2.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem2.loc18, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1 = specific_constant imports.%Core.Greater.a10, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1]
// CHECK:STDOUT:   %Greater.ref.loc18: %Cpp.long_long.as.OrderedWith.impl.Greater.type.2f531e.1 = name_ref Greater, %.loc18_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b790ce.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc18: <bound method> = bound_method %a.ref.loc18, %Greater.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long_long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_5.5: %Cpp.long_long = converted %int_1.loc18, %.loc18_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc18: <specific function> = specific_function %Greater.ref.loc18, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.b9662b.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_7: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_7: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_7: init %Cpp.long_long = call %bound_method.loc18_7(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_7.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_7.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.47c = impl_witness_access constants.%OrderedWith.impl_witness.31a, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1 = specific_constant imports.%Core.Less.511, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1]
// CHECK:STDOUT:   %Less.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Less.type.ac9e3b.1 = name_ref Less, %.loc19_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.26cbb1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Less.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long_long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_5.5: %Cpp.long_long = converted %int_1.loc19, %.loc19_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc19: <specific function> = specific_function %Less.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.50db42.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc20: %.f68 = impl_witness_access constants.%OrderedWith.impl_witness.31a, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem3.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem3.loc20, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1 = specific_constant imports.%Core.GreaterOrEquivalent.489, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.23f245.1 = name_ref GreaterOrEquivalent, %.loc20_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1b92ea.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc20: <bound method> = bound_method %a.ref.loc20, %GreaterOrEquivalent.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long_long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_5.5: %Cpp.long_long = converted %int_1.loc20, %.loc20_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc20: <specific function> = specific_function %GreaterOrEquivalent.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.662ddf.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_8: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_8: init %Cpp.long_long = call %bound_method.loc20_8(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_8.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_8.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc21: %.c1a = impl_witness_access constants.%OrderedWith.impl_witness.31a, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem1.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.82e) [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.a03 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.d52]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1 = specific_constant imports.%Core.LessOrEquivalent.aaf, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.e236e2.1 = name_ref LessOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.50ecf5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %LessOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long_long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_5.5: %Cpp.long_long = converted %int_1.loc21, %.loc21_5.4 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc21: <specific function> = specific_function %LessOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.d52) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b1c4c5.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc23_10: type = splice_block %i64.loc23 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc23: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc23: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc23: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc23_16.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_16.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i64 = call %bound_method.loc23_16.2(%int_1.loc23) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc23_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc23_16.2: %i64 = converted %int_1.loc23, %.loc23_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc23_16.2
// CHECK:STDOUT:   %a.ref.loc24: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc24: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc24_5.1: %.6ba = impl_witness_access constants.%EqWith.impl_witness.756, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.2]
// CHECK:STDOUT:   %bound_method.loc24_5.1: <bound method> = bound_method %a.ref.loc24, %impl.elem0.loc24_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc24_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc24_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc24_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc24_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc24_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc24_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_5.1, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.1]
// CHECK:STDOUT:   %bound_method.loc24_5.2: <bound method> = bound_method %a.ref.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_5.3: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = specific_constant imports.%Core.Equal.357, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Equal.ref.loc24: %Cpp.long_long.as.EqWith.impl.Equal.type.2f62f8.1 = name_ref Equal, %.loc24_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.610927.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc24: <bound method> = bound_method %a.ref.loc24, %Equal.ref.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_5.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_5.3: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc24_5: init %Cpp.long_long = call %bound_method.loc24_5.3(%b.ref.loc24)
// CHECK:STDOUT:   %.loc24_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc24_5
// CHECK:STDOUT:   %.loc24_5.5: %Cpp.long_long = converted %b.ref.loc24, %.loc24_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc24: <specific function> = specific_function %Equal.ref.loc24, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.29a235.2]
// CHECK:STDOUT:   %bound_method.loc24_5.4: <bound method> = bound_method %a.ref.loc24, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_8: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_8: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc24_8: init %Cpp.long_long = call %bound_method.loc24_8(%b.ref.loc24)
// CHECK:STDOUT:   %.loc24_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc24_8
// CHECK:STDOUT:   %.loc24_8.2: %Cpp.long_long = converted %b.ref.loc24, %.loc24_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc24: init bool = call %bound_method.loc24_5.4(%a.ref.loc24, %.loc24_8.2)
// CHECK:STDOUT:   %a.ref.loc25: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc25: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc25: %.838 = impl_witness_access constants.%EqWith.impl_witness.756, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.2]
// CHECK:STDOUT:   %bound_method.loc25_5.1: <bound method> = bound_method %a.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc25_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc25_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc25_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc25_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.1]
// CHECK:STDOUT:   %bound_method.loc25_5.2: <bound method> = bound_method %a.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_5.3: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = specific_constant imports.%Core.NotEqual.dea, @Cpp.long_long.as.EqWith.impl.1bc(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %NotEqual.ref.loc25: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8239fe.1 = name_ref NotEqual, %.loc25_5.3 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.81fbd8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc25: <bound method> = bound_method %a.ref.loc25, %NotEqual.ref.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_5.3: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc25_5: init %Cpp.long_long = call %bound_method.loc25_5.3(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc25_5
// CHECK:STDOUT:   %.loc25_5.5: %Cpp.long_long = converted %b.ref.loc25, %.loc25_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc25: <specific function> = specific_function %NotEqual.ref.loc25, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.b9c577.2]
// CHECK:STDOUT:   %bound_method.loc25_5.4: <bound method> = bound_method %a.ref.loc25, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_8: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_8: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc25_8: init %Cpp.long_long = call %bound_method.loc25_8(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc25_8
// CHECK:STDOUT:   %.loc25_8.2: %Cpp.long_long = converted %b.ref.loc25, %.loc25_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc25: init bool = call %bound_method.loc25_5.4(%a.ref.loc25, %.loc25_8.2)
// CHECK:STDOUT:   %a.ref.loc26: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc26: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc26: %.1fe = impl_witness_access constants.%OrderedWith.impl_witness.940, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.2]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %a.ref.loc26, %impl.elem2.loc26
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc26_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc26_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc26_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc26_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem2.loc26, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.1]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %a.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_5.3: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = specific_constant imports.%Core.Greater.a10, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Greater.ref.loc26: %Cpp.long_long.as.OrderedWith.impl.Greater.type.1d3e50.1 = name_ref Greater, %.loc26_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.136328.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc26: <bound method> = bound_method %a.ref.loc26, %Greater.ref.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_5.3: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc26_5: init %Cpp.long_long = call %bound_method.loc26_5.3(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc26_5
// CHECK:STDOUT:   %.loc26_5.5: %Cpp.long_long = converted %b.ref.loc26, %.loc26_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc26: <specific function> = specific_function %Greater.ref.loc26, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.9b8255.2]
// CHECK:STDOUT:   %bound_method.loc26_5.4: <bound method> = bound_method %a.ref.loc26, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_7: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_7: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc26_7: init %Cpp.long_long = call %bound_method.loc26_7(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc26_7
// CHECK:STDOUT:   %.loc26_7.2: %Cpp.long_long = converted %b.ref.loc26, %.loc26_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc26: init bool = call %bound_method.loc26_5.4(%a.ref.loc26, %.loc26_7.2)
// CHECK:STDOUT:   %a.ref.loc27: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc27: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc27_5.1: %.0e7 = impl_witness_access constants.%OrderedWith.impl_witness.940, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.2]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %a.ref.loc27, %impl.elem0.loc27_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc27_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc27_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc27_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc27_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27_5.1, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.1]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %a.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_5.3: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = specific_constant imports.%Core.Less.511, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Less.ref.loc27: %Cpp.long_long.as.OrderedWith.impl.Less.type.c81272.1 = name_ref Less, %.loc27_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.39610d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc27: <bound method> = bound_method %a.ref.loc27, %Less.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_5.2: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_5.3: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_5.2
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc27_5: init %Cpp.long_long = call %bound_method.loc27_5.3(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc27_5
// CHECK:STDOUT:   %.loc27_5.5: %Cpp.long_long = converted %b.ref.loc27, %.loc27_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc27: <specific function> = specific_function %Less.ref.loc27, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.294c7c.2]
// CHECK:STDOUT:   %bound_method.loc27_5.4: <bound method> = bound_method %a.ref.loc27, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_7: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_7: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc27_7: init %Cpp.long_long = call %bound_method.loc27_7(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc27_7
// CHECK:STDOUT:   %.loc27_7.2: %Cpp.long_long = converted %b.ref.loc27, %.loc27_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc27: init bool = call %bound_method.loc27_5.4(%a.ref.loc27, %.loc27_7.2)
// CHECK:STDOUT:   %a.ref.loc28: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc28: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc28: %.30d = impl_witness_access constants.%OrderedWith.impl_witness.940, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.2]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %a.ref.loc28, %impl.elem3.loc28
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc28_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc28_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc28_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc28_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem3.loc28, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.1]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %a.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_5.3: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = specific_constant imports.%Core.GreaterOrEquivalent.489, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc28: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.097dad.1 = name_ref GreaterOrEquivalent, %.loc28_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.777234.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc28: <bound method> = bound_method %a.ref.loc28, %GreaterOrEquivalent.ref.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_5.3: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc28_5: init %Cpp.long_long = call %bound_method.loc28_5.3(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc28_5
// CHECK:STDOUT:   %.loc28_5.5: %Cpp.long_long = converted %b.ref.loc28, %.loc28_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28: <specific function> = specific_function %GreaterOrEquivalent.ref.loc28, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ffe884.2]
// CHECK:STDOUT:   %bound_method.loc28_5.4: <bound method> = bound_method %a.ref.loc28, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_8: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_8: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc28_8: init %Cpp.long_long = call %bound_method.loc28_8(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc28_8
// CHECK:STDOUT:   %.loc28_8.2: %Cpp.long_long = converted %b.ref.loc28, %.loc28_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc28: init bool = call %bound_method.loc28_5.4(%a.ref.loc28, %.loc28_8.2)
// CHECK:STDOUT:   %a.ref.loc29: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc29: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc29: %.90d = impl_witness_access constants.%OrderedWith.impl_witness.940, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.2]
// CHECK:STDOUT:   %bound_method.loc29_5.1: <bound method> = bound_method %a.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_5.1: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc29_5.1: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc29_5.1 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_5.2: %ImplicitAs.type.a03 = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.1b3) [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %.loc29_5.2: %ImplicitAs.type.a03 = converted constants.%i64, %ImplicitAs.facet.loc29_5.2 [concrete = constants.%ImplicitAs.facet.c59]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.1]
// CHECK:STDOUT:   %bound_method.loc29_5.2: <bound method> = bound_method %a.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_5.3: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = specific_constant imports.%Core.LessOrEquivalent.aaf, @Cpp.long_long.as.OrderedWith.impl.ef6(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc29: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.82788f.1 = name_ref LessOrEquivalent, %.loc29_5.3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.29b20a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc29: <bound method> = bound_method %a.ref.loc29, %LessOrEquivalent.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_5: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_5.3: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_5
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc29_5: init %Cpp.long_long = call %bound_method.loc29_5.3(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_5.4: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc29_5
// CHECK:STDOUT:   %.loc29_5.5: %Cpp.long_long = converted %b.ref.loc29, %.loc29_5.4
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29: <specific function> = specific_function %LessOrEquivalent.ref.loc29, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c59) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.5291ea.2]
// CHECK:STDOUT:   %bound_method.loc29_5.4: <bound method> = bound_method %a.ref.loc29, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_8: %.b29 = impl_witness_access constants.%ImplicitAs.impl_witness.1b3, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_8: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc29_8: init %Cpp.long_long = call %bound_method.loc29_8(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc29_8
// CHECK:STDOUT:   %.loc29_8.2: %Cpp.long_long = converted %b.ref.loc29, %.loc29_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc29: init bool = call %bound_method.loc29_5.4(%a.ref.loc29, %.loc29_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a03: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6e4: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.896 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d52: %ImplicitAs.type.a03 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.82e) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %ImplicitAs.Convert.type.6e4, %ImplicitAs.facet.d52 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a20 [concrete]
// CHECK:STDOUT:   %int_1.092: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %.aba: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.4be: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.6d5: type = fn_type @EqWith.Equal, @EqWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.c18: type = fn_type @EqWith.NotEqual, @EqWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %T.ea5: %ImplicitAs.type.a03 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.3, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.4, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.3, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bbf235.1: %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.4, @T.binding.as_type.as.EqWith.impl.b07(%T.ea5) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bbf235.2: %T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.39a54f.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %T.354: %ImplicitAs.type.39a54f.1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.db1e05.1: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.42c(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.088a98.1: %Int.as.EqWith.impl.NotEqual.type.db1e05.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.db1e05.2: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.42c(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.088a98.2: %Int.as.EqWith.impl.NotEqual.type.db1e05.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.a94820.1: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.42c(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.737ae8.1: %Int.as.EqWith.impl.Equal.type.a94820.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.a94820.2: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.42c(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.737ae8.2: %Int.as.EqWith.impl.Equal.type.a94820.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a63: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.563 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.660: %ImplicitAs.type.2ad = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.a63) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.367: <witness> = impl_witness imports.%EqWith.impl_witness_table.600, @Int.as.EqWith.impl.42c(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.e5f20b.1: type = fn_type @Int.as.EqWith.impl.Equal.3, @Int.as.EqWith.impl.42c(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.e49d18.1: %Int.as.EqWith.impl.Equal.type.e5f20b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.9ab064.1: type = fn_type @Int.as.EqWith.impl.NotEqual.3, @Int.as.EqWith.impl.42c(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.af2dce.1: %Int.as.EqWith.impl.NotEqual.type.9ab064.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.type.e5f20b.2: type = fn_type @Int.as.EqWith.impl.Equal.2, @Int.as.EqWith.impl.42c(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.e49d18.2: %Int.as.EqWith.impl.Equal.type.e5f20b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.type.9ab064.2: type = fn_type @Int.as.EqWith.impl.NotEqual.2, @Int.as.EqWith.impl.42c(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.af2dce.2: %Int.as.EqWith.impl.NotEqual.type.9ab064.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.5fc: %EqWith.type.4be = facet_value %i64, (%EqWith.impl_witness.367) [concrete]
// CHECK:STDOUT:   %.144: type = fn_type_with_self_type %EqWith.Equal.type.6d5, %EqWith.facet.5fc [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound.e9a871.1: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.Equal.e49d18.2 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.443739.1: <specific function> = specific_function %Int.as.EqWith.impl.Equal.e49d18.2, @Int.as.EqWith.impl.Equal.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.a45aed.1: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.Equal.specific_fn.443739.1 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound.e9a871.2: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.Equal.e49d18.1 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.443739.2: <specific function> = specific_function %Int.as.EqWith.impl.Equal.e49d18.1, @Int.as.EqWith.impl.Equal.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.a45aed.2: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.Equal.specific_fn.443739.2 [concrete]
// CHECK:STDOUT:   %.912: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.660 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b78: type = fn_type_with_self_type %EqWith.NotEqual.type.c18, %EqWith.facet.5fc [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound.9611a1.1: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.NotEqual.af2dce.2 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.044259.1: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.af2dce.2, @Int.as.EqWith.impl.NotEqual.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.356533.1: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.NotEqual.specific_fn.044259.1 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound.9611a1.2: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.NotEqual.af2dce.1 [concrete]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.044259.2: <specific function> = specific_function %Int.as.EqWith.impl.NotEqual.af2dce.1, @Int.as.EqWith.impl.NotEqual.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.356533.2: <bound method> = bound_method %int_1.41a, %Int.as.EqWith.impl.NotEqual.specific_fn.044259.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.65e: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.bc1: type = fn_type @OrderedWith.Less, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.1cc: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.921: type = fn_type @OrderedWith.Greater, @OrderedWith(%Cpp.long_long) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.fcf: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%Cpp.long_long) [concrete]
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
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.abd089.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.abd089.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.8b8cf7.1: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.4a2789.1: %Int.as.OrderedWith.impl.Greater.type.8b8cf7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.8b8cf7.2: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.4a2789.2: %Int.as.OrderedWith.impl.Greater.type.8b8cf7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.52f78e.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.52f78e.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.797264.1: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.c4a430.1: %Int.as.OrderedWith.impl.Less.type.797264.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.797264.2: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.aba(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.c4a430.2: %Int.as.OrderedWith.impl.Less.type.797264.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.27d: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.03f, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.0fa05d.1: type = fn_type @Int.as.OrderedWith.impl.Less.3, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.f7d683.1: %Int.as.OrderedWith.impl.Less.type.0fa05d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.3, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.e91ea2.1: type = fn_type @Int.as.OrderedWith.impl.Greater.3, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.2b7c26.1: %Int.as.OrderedWith.impl.Greater.type.e91ea2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.3, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.0fa05d.2: type = fn_type @Int.as.OrderedWith.impl.Less.2, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.f7d683.2: %Int.as.OrderedWith.impl.Less.type.0fa05d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.2: type = fn_type @Int.as.OrderedWith.impl.LessOrEquivalent.2, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.2: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.type.e91ea2.2: type = fn_type @Int.as.OrderedWith.impl.Greater.2, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.2b7c26.2: %Int.as.OrderedWith.impl.Greater.type.e91ea2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.2: type = fn_type @Int.as.OrderedWith.impl.GreaterOrEquivalent.2, @Int.as.OrderedWith.impl.aba(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.2: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.642: %OrderedWith.type.65e = facet_value %i64, (%OrderedWith.impl_witness.27d) [concrete]
// CHECK:STDOUT:   %.8f1: type = fn_type_with_self_type %OrderedWith.Greater.type.921, %OrderedWith.facet.642 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound.d9ebd2.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Greater.2b7c26.2 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.2b7c26.2, @Int.as.OrderedWith.impl.Greater.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.8ea317.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound.d9ebd2.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Greater.2b7c26.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Greater.2b7c26.1, @Int.as.OrderedWith.impl.Greater.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.8ea317.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.2 [concrete]
// CHECK:STDOUT:   %.6ba: type = fn_type_with_self_type %OrderedWith.Less.type.bc1, %OrderedWith.facet.642 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound.d163a5.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Less.f7d683.2 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.e635f0.1: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.f7d683.2, @Int.as.OrderedWith.impl.Less.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.af4a52.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Less.specific_fn.e635f0.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound.d163a5.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Less.f7d683.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.e635f0.2: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.f7d683.1, @Int.as.OrderedWith.impl.Less.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.af4a52.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.Less.specific_fn.e635f0.2 [concrete]
// CHECK:STDOUT:   %.cf1: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.fcf, %OrderedWith.facet.642 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.e6a2af.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.2 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.1: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.2, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.e2b9a3.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.e6a2af.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.2: <specific function> = specific_function %Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.e2b9a3.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.2 [concrete]
// CHECK:STDOUT:   %.ec7: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.1cc, %OrderedWith.facet.642 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound.2ed161.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.2 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.1: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.2, @Int.as.OrderedWith.impl.LessOrEquivalent.2(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.cb16a7.1: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound.2ed161.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1 [concrete]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.2: <specific function> = specific_function %Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1, @Int.as.OrderedWith.impl.LessOrEquivalent.3(%int_64, %ImplicitAs.facet.660) [concrete]
// CHECK:STDOUT:   %bound_method.cb16a7.2: <bound method> = bound_method %int_1.41a, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.2 [concrete]
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
// CHECK:STDOUT:   %.90a: type = fn_type_with_self_type %EqWith.Equal.type.6d5, %EqWith.facet.db2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2, @T.binding.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.1699f5.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1, @T.binding.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.1699f5.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2 [concrete]
// CHECK:STDOUT:   %.d05: type = fn_type_with_self_type %EqWith.NotEqual.type.c18, %EqWith.facet.db2 [concrete]
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
// CHECK:STDOUT:   %.5c6: type = fn_type_with_self_type %OrderedWith.Greater.type.921, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2, @T.binding.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.a32da1.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1, @T.binding.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.a32da1.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2 [concrete]
// CHECK:STDOUT:   %.f13: type = fn_type_with_self_type %OrderedWith.Less.type.bc1, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2, @T.binding.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.8f808d.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1, @T.binding.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.8f808d.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2 [concrete]
// CHECK:STDOUT:   %.184: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.fcf, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.2c548f.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.2c548f.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2 [concrete]
// CHECK:STDOUT:   %.5f9: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.1cc, %OrderedWith.facet.88a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.caef99.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.d52) [concrete]
// CHECK:STDOUT:   %bound_method.caef99.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.288: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d64: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aba = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.896 = impl_witness_table (%Core.import_ref.d64), @Core.IntLiteral.as.ImplicitAs.impl.a26 [concrete]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eb5: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.type.2 (%T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.2 (constants.%T.binding.as_type.as.EqWith.impl.Equal.bbf235.1)]
// CHECK:STDOUT:   %Core.import_ref.597: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.type.2 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.7e5 = impl_witness_table (%Core.import_ref.eb5, %Core.import_ref.597), @T.binding.as_type.as.EqWith.impl.b07 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.421: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.type.1 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.d96a78.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.83a9a5.2)]
// CHECK:STDOUT:   %Core.Equal.d94: @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.type.1 (%T.binding.as_type.as.EqWith.impl.Equal.type.066d1d.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.b07.%T.binding.as_type.as.EqWith.impl.Equal.1 (constants.%T.binding.as_type.as.EqWith.impl.Equal.bbf235.2)]
// CHECK:STDOUT:   %Core.import_ref.613: @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.Equal.type.2 (%Int.as.EqWith.impl.Equal.type.a94820.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.Equal.2 (constants.%Int.as.EqWith.impl.Equal.737ae8.1)]
// CHECK:STDOUT:   %Core.import_ref.23d: @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.NotEqual.type.2 (%Int.as.EqWith.impl.NotEqual.type.db1e05.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.NotEqual.2 (constants.%Int.as.EqWith.impl.NotEqual.088a98.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.600 = impl_witness_table (%Core.import_ref.613, %Core.import_ref.23d), @Int.as.EqWith.impl.42c [concrete]
// CHECK:STDOUT:   %Core.NotEqual.334: @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.NotEqual.type.1 (%Int.as.EqWith.impl.NotEqual.type.db1e05.2) = import_ref Core//prelude/types/int, NotEqual, loaded [symbolic = @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.NotEqual.1 (constants.%Int.as.EqWith.impl.NotEqual.088a98.2)]
// CHECK:STDOUT:   %Core.Equal.ca9: @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.Equal.type.1 (%Int.as.EqWith.impl.Equal.type.a94820.2) = import_ref Core//prelude/types/int, Equal, loaded [symbolic = @Int.as.EqWith.impl.42c.%Int.as.EqWith.impl.Equal.1 (constants.%Int.as.EqWith.impl.Equal.737ae8.2)]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.91c: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.563 = impl_witness_table (%Core.import_ref.91c), @Cpp.long_long.as.ImplicitAs.impl.034 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d4b: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.type.2 (%T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.31696a.1)]
// CHECK:STDOUT:   %Core.import_ref.54d: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.1)]
// CHECK:STDOUT:   %Core.import_ref.aa8: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.type.2 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.1)]
// CHECK:STDOUT:   %Core.import_ref.240: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.8cd = impl_witness_table (%Core.import_ref.d4b, %Core.import_ref.54d, %Core.import_ref.aa8, %Core.import_ref.240), @T.binding.as_type.as.OrderedWith.impl.895 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.b91: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.b7b4dd.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.06285b.2)]
// CHECK:STDOUT:   %Core.Greater.cff: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.type.1 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.1733df.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.07eebf.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.792: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.bfda3a.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.d63604.2)]
// CHECK:STDOUT:   %Core.Less.aac: @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.type.1 (%T.binding.as_type.as.OrderedWith.impl.Less.type.4623e2.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.895.%T.binding.as_type.as.OrderedWith.impl.Less.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.31696a.2)]
// CHECK:STDOUT:   %Core.import_ref.711: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Less.type.2 (%Int.as.OrderedWith.impl.Less.type.797264.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Less.2 (constants.%Int.as.OrderedWith.impl.Less.c4a430.1)]
// CHECK:STDOUT:   %Core.import_ref.0b3: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.52f78e.1)]
// CHECK:STDOUT:   %Core.import_ref.d5e: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Greater.type.2 (%Int.as.OrderedWith.impl.Greater.type.8b8cf7.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Greater.2 (constants.%Int.as.OrderedWith.impl.Greater.4a2789.1)]
// CHECK:STDOUT:   %Core.import_ref.a06: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.abd089.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.03f = impl_witness_table (%Core.import_ref.711, %Core.import_ref.0b3, %Core.import_ref.d5e, %Core.import_ref.a06), @Int.as.OrderedWith.impl.aba [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.757: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Int.as.OrderedWith.impl.GreaterOrEquivalent.type.40683b.2) = import_ref Core//prelude/types/int, GreaterOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.abd089.2)]
// CHECK:STDOUT:   %Core.Greater.cd9: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Greater.type.1 (%Int.as.OrderedWith.impl.Greater.type.8b8cf7.2) = import_ref Core//prelude/types/int, Greater, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Greater.1 (constants.%Int.as.OrderedWith.impl.Greater.4a2789.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.c5f: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Int.as.OrderedWith.impl.LessOrEquivalent.type.2a47fa.2) = import_ref Core//prelude/types/int, LessOrEquivalent, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Int.as.OrderedWith.impl.LessOrEquivalent.52f78e.2)]
// CHECK:STDOUT:   %Core.Less.df4: @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Less.type.1 (%Int.as.OrderedWith.impl.Less.type.797264.2) = import_ref Core//prelude/types/int, Less, loaded [symbolic = @Int.as.OrderedWith.impl.aba.%Int.as.OrderedWith.impl.Less.1 (constants.%Int.as.OrderedWith.impl.Less.c4a430.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.76e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc8_26.2
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc9_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc9_6.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9_6: <specific function> = specific_function %impl.elem0.loc9_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_6.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_6.2(%int_1.loc9) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc9 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc9_6.2: %i64 = converted %int_1.loc9, %.loc9_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc9: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9_14: %.144 = impl_witness_access constants.%EqWith.impl_witness.367, element0 [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.2]
// CHECK:STDOUT:   %bound_method.loc9_14.1: <bound method> = bound_method %.loc9_6.2, %impl.elem0.loc9_14 [concrete = constants.%Int.as.EqWith.impl.Equal.bound.e9a871.1]
// CHECK:STDOUT:   %specific_fn.loc9_14: <specific function> = specific_function %impl.elem0.loc9_14, @Int.as.EqWith.impl.Equal.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.443739.1]
// CHECK:STDOUT:   %bound_method.loc9_14.2: <bound method> = bound_method %.loc9_6.2, %specific_fn.loc9_14 [concrete = constants.%bound_method.a45aed.1]
// CHECK:STDOUT:   %.loc9_14: %Int.as.EqWith.impl.Equal.type.e5f20b.1 = specific_constant imports.%Core.Equal.ca9, @Int.as.EqWith.impl.42c(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.1]
// CHECK:STDOUT:   %Equal.ref.loc9: %Int.as.EqWith.impl.Equal.type.e5f20b.1 = name_ref Equal, %.loc9_14 [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound.loc9: <bound method> = bound_method %.loc9_6.2, %Equal.ref.loc9 [concrete = constants.%Int.as.EqWith.impl.Equal.bound.e9a871.2]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.loc9: <specific function> = specific_function %Equal.ref.loc9, @Int.as.EqWith.impl.Equal.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.443739.2]
// CHECK:STDOUT:   %bound_method.loc9_14.3: <bound method> = bound_method %.loc9_6.2, %Int.as.EqWith.impl.Equal.specific_fn.loc9 [concrete = constants.%bound_method.a45aed.2]
// CHECK:STDOUT:   %impl.elem0.loc9_17: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_17: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9_17
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_17(%a.ref.loc9)
// CHECK:STDOUT:   %.loc9_17.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %.loc9_17.2: %i64 = converted %a.ref.loc9, %.loc9_17.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call.loc9: init bool = call %bound_method.loc9_14.3(%.loc9_6.2, %.loc9_17.2)
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc10: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc10: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc10_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc10_6.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10_6: <specific function> = specific_function %impl.elem0.loc10_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_6.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_6.2(%int_1.loc10) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc10_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc10 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc10_6.2: %i64 = converted %int_1.loc10, %.loc10_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc10: %.b78 = impl_witness_access constants.%EqWith.impl_witness.367, element1 [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.2]
// CHECK:STDOUT:   %bound_method.loc10_14.1: <bound method> = bound_method %.loc10_6.2, %impl.elem1.loc10 [concrete = constants.%Int.as.EqWith.impl.NotEqual.bound.9611a1.1]
// CHECK:STDOUT:   %specific_fn.loc10_14: <specific function> = specific_function %impl.elem1.loc10, @Int.as.EqWith.impl.NotEqual.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.044259.1]
// CHECK:STDOUT:   %bound_method.loc10_14.2: <bound method> = bound_method %.loc10_6.2, %specific_fn.loc10_14 [concrete = constants.%bound_method.356533.1]
// CHECK:STDOUT:   %.loc10_14: %Int.as.EqWith.impl.NotEqual.type.9ab064.1 = specific_constant imports.%Core.NotEqual.334, @Int.as.EqWith.impl.42c(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.1]
// CHECK:STDOUT:   %NotEqual.ref.loc10: %Int.as.EqWith.impl.NotEqual.type.9ab064.1 = name_ref NotEqual, %.loc10_14 [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound.loc10: <bound method> = bound_method %.loc10_6.2, %NotEqual.ref.loc10 [concrete = constants.%Int.as.EqWith.impl.NotEqual.bound.9611a1.2]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.loc10: <specific function> = specific_function %NotEqual.ref.loc10, @Int.as.EqWith.impl.NotEqual.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.044259.2]
// CHECK:STDOUT:   %bound_method.loc10_14.3: <bound method> = bound_method %.loc10_6.2, %Int.as.EqWith.impl.NotEqual.specific_fn.loc10 [concrete = constants.%bound_method.356533.2]
// CHECK:STDOUT:   %impl.elem0.loc10_17: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_17: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_17
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc10: init %i64 = call %bound_method.loc10_17(%a.ref.loc10)
// CHECK:STDOUT:   %.loc10_17.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_17.2: %i64 = converted %a.ref.loc10, %.loc10_17.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call.loc10: init bool = call %bound_method.loc10_14.3(%.loc10_6.2, %.loc10_17.2)
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc11: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc11: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc11_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc11_6.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_6: <specific function> = specific_function %impl.elem0.loc11_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_6.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_6.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_6.2: %i64 = converted %int_1.loc11, %.loc11_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc11: %.8f1 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element2 [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.2]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %.loc11_6.2, %impl.elem2.loc11 [concrete = constants.%Int.as.OrderedWith.impl.Greater.bound.d9ebd2.1]
// CHECK:STDOUT:   %specific_fn.loc11_14: <specific function> = specific_function %impl.elem2.loc11, @Int.as.OrderedWith.impl.Greater.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.1]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %.loc11_6.2, %specific_fn.loc11_14 [concrete = constants.%bound_method.8ea317.1]
// CHECK:STDOUT:   %.loc11_14: %Int.as.OrderedWith.impl.Greater.type.e91ea2.1 = specific_constant imports.%Core.Greater.cd9, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.1]
// CHECK:STDOUT:   %Greater.ref.loc11: %Int.as.OrderedWith.impl.Greater.type.e91ea2.1 = name_ref Greater, %.loc11_14 [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound.loc11: <bound method> = bound_method %.loc11_6.2, %Greater.ref.loc11 [concrete = constants.%Int.as.OrderedWith.impl.Greater.bound.d9ebd2.2]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.loc11: <specific function> = specific_function %Greater.ref.loc11, @Int.as.OrderedWith.impl.Greater.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.2]
// CHECK:STDOUT:   %bound_method.loc11_14.3: <bound method> = bound_method %.loc11_6.2, %Int.as.OrderedWith.impl.Greater.specific_fn.loc11 [concrete = constants.%bound_method.8ea317.2]
// CHECK:STDOUT:   %impl.elem0.loc11_16: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_16: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_16
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_16(%a.ref.loc11)
// CHECK:STDOUT:   %.loc11_16.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_16.2: %i64 = converted %a.ref.loc11, %.loc11_16.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call.loc11: init bool = call %bound_method.loc11_14.3(%.loc11_6.2, %.loc11_16.2)
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc12: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc12: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc12_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc12_6.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc12_6: <specific function> = specific_function %impl.elem0.loc12_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_6.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_6.2(%int_1.loc12) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc12_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc12 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc12_6.2: %i64 = converted %int_1.loc12, %.loc12_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc12_14: %.6ba = impl_witness_access constants.%OrderedWith.impl_witness.27d, element0 [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.2]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %.loc12_6.2, %impl.elem0.loc12_14 [concrete = constants.%Int.as.OrderedWith.impl.Less.bound.d163a5.1]
// CHECK:STDOUT:   %specific_fn.loc12_14: <specific function> = specific_function %impl.elem0.loc12_14, @Int.as.OrderedWith.impl.Less.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.e635f0.1]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %.loc12_6.2, %specific_fn.loc12_14 [concrete = constants.%bound_method.af4a52.1]
// CHECK:STDOUT:   %.loc12_14: %Int.as.OrderedWith.impl.Less.type.0fa05d.1 = specific_constant imports.%Core.Less.df4, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.1]
// CHECK:STDOUT:   %Less.ref.loc12: %Int.as.OrderedWith.impl.Less.type.0fa05d.1 = name_ref Less, %.loc12_14 [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound.loc12: <bound method> = bound_method %.loc12_6.2, %Less.ref.loc12 [concrete = constants.%Int.as.OrderedWith.impl.Less.bound.d163a5.2]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.loc12: <specific function> = specific_function %Less.ref.loc12, @Int.as.OrderedWith.impl.Less.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.e635f0.2]
// CHECK:STDOUT:   %bound_method.loc12_14.3: <bound method> = bound_method %.loc12_6.2, %Int.as.OrderedWith.impl.Less.specific_fn.loc12 [concrete = constants.%bound_method.af4a52.2]
// CHECK:STDOUT:   %impl.elem0.loc12_16: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_16: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_16
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_16(%a.ref.loc12)
// CHECK:STDOUT:   %.loc12_16.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_16.2: %i64 = converted %a.ref.loc12, %.loc12_16.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call.loc12: init bool = call %bound_method.loc12_14.3(%.loc12_6.2, %.loc12_16.2)
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc13: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc13: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc13_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc13_6.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13_6: <specific function> = specific_function %impl.elem0.loc13_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_6.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_6.2(%int_1.loc13) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc13_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc13 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc13_6.2: %i64 = converted %int_1.loc13, %.loc13_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc13: %.cf1 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element3 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.2]
// CHECK:STDOUT:   %bound_method.loc13_14.1: <bound method> = bound_method %.loc13_6.2, %impl.elem3.loc13 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.e6a2af.1]
// CHECK:STDOUT:   %specific_fn.loc13_14: <specific function> = specific_function %impl.elem3.loc13, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.1]
// CHECK:STDOUT:   %bound_method.loc13_14.2: <bound method> = bound_method %.loc13_6.2, %specific_fn.loc13_14 [concrete = constants.%bound_method.e2b9a3.1]
// CHECK:STDOUT:   %.loc13_14: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1 = specific_constant imports.%Core.GreaterOrEquivalent.757, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc13: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1 = name_ref GreaterOrEquivalent, %.loc13_14 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc13: <bound method> = bound_method %.loc13_6.2, %GreaterOrEquivalent.ref.loc13 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.e6a2af.2]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc13: <specific function> = specific_function %GreaterOrEquivalent.ref.loc13, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.2]
// CHECK:STDOUT:   %bound_method.loc13_14.3: <bound method> = bound_method %.loc13_6.2, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc13 [concrete = constants.%bound_method.e2b9a3.2]
// CHECK:STDOUT:   %impl.elem0.loc13_17: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_17: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_17
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_17(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_17.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_17.2: %i64 = converted %a.ref.loc13, %.loc13_17.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.call.loc13: init bool = call %bound_method.loc13_14.3(%.loc13_6.2, %.loc13_17.2)
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc14: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc14: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc14_6: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc14_6.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_6 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc14_6: <specific function> = specific_function %impl.elem0.loc14_6, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_6.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_6 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_6.2(%int_1.loc14) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc14_6.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc14_6.2: %i64 = converted %int_1.loc14, %.loc14_6.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc14: %.ec7 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element1 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.2]
// CHECK:STDOUT:   %bound_method.loc14_14.1: <bound method> = bound_method %.loc14_6.2, %impl.elem1.loc14 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.bound.2ed161.1]
// CHECK:STDOUT:   %specific_fn.loc14_14: <specific function> = specific_function %impl.elem1.loc14, @Int.as.OrderedWith.impl.LessOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.1]
// CHECK:STDOUT:   %bound_method.loc14_14.2: <bound method> = bound_method %.loc14_6.2, %specific_fn.loc14_14 [concrete = constants.%bound_method.cb16a7.1]
// CHECK:STDOUT:   %.loc14_14: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1 = specific_constant imports.%Core.LessOrEquivalent.c5f, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc14: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1 = name_ref LessOrEquivalent, %.loc14_14 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound.loc14: <bound method> = bound_method %.loc14_6.2, %LessOrEquivalent.ref.loc14 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.bound.2ed161.2]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc14: <specific function> = specific_function %LessOrEquivalent.ref.loc14, @Int.as.OrderedWith.impl.LessOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.2]
// CHECK:STDOUT:   %bound_method.loc14_14.3: <bound method> = bound_method %.loc14_6.2, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc14 [concrete = constants.%bound_method.cb16a7.2]
// CHECK:STDOUT:   %impl.elem0.loc14_17: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_17: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_17
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_17(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_17.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_17.2: %i64 = converted %a.ref.loc14, %.loc14_17.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call.loc14: init bool = call %bound_method.loc14_14.3(%.loc14_6.2, %.loc14_17.2)
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc16: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.90a = impl_witness_access constants.%EqWith.impl_witness.a59, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.1]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @T.binding.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.1699f5.1]
// CHECK:STDOUT:   %.loc16_5: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1 = specific_constant imports.%Core.Equal.d94, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1]
// CHECK:STDOUT:   %Equal.ref.loc16: %T.binding.as_type.as.EqWith.impl.Equal.type.e71550.1 = name_ref Equal, %.loc16_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.3cf73d.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound: <bound method> = bound_method %int_1.loc16, %Equal.ref.loc16 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.e0853e.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn: <specific function> = specific_function %Equal.ref.loc16, @T.binding.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.54d028.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %int_1.loc16, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn [concrete = constants.%bound_method.1699f5.2]
// CHECK:STDOUT:   %impl.elem0.loc16_3: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc16_3: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long_long = call %bound_method.loc16_3(%int_1.loc16) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc16_3.2: %Cpp.long_long = converted %int_1.loc16, %.loc16_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc16_5.3(%.loc16_3.2, %a.ref.loc16)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc17: %.d05 = impl_witness_access constants.%EqWith.impl_witness.a59, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem1.loc17 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem1.loc17, @T.binding.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.7aede5.1]
// CHECK:STDOUT:   %.loc17_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1 = specific_constant imports.%Core.NotEqual.421, @T.binding.as_type.as.EqWith.impl.b07(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1]
// CHECK:STDOUT:   %NotEqual.ref.loc17: %T.binding.as_type.as.EqWith.impl.NotEqual.type.635206.1 = name_ref NotEqual, %.loc17_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.55dccd.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound: <bound method> = bound_method %int_1.loc17, %NotEqual.ref.loc17 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.0bb6c5.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn: <specific function> = specific_function %NotEqual.ref.loc17, @T.binding.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.d399fc.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn [concrete = constants.%bound_method.7aede5.2]
// CHECK:STDOUT:   %impl.elem0.loc17: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc18: %.5c6 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem2.loc18 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem2.loc18, @T.binding.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.a32da1.1]
// CHECK:STDOUT:   %.loc18_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1 = specific_constant imports.%Core.Greater.cff, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1]
// CHECK:STDOUT:   %Greater.ref.loc18: %T.binding.as_type.as.OrderedWith.impl.Greater.type.e76509.1 = name_ref Greater, %.loc18_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.9aafa8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound: <bound method> = bound_method %int_1.loc18, %Greater.ref.loc18 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.483e05.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn: <specific function> = specific_function %Greater.ref.loc18, @T.binding.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.d2b07b.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn [concrete = constants.%bound_method.a32da1.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.f13 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @T.binding.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.8f808d.1]
// CHECK:STDOUT:   %.loc19_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1 = specific_constant imports.%Core.Less.aac, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1]
// CHECK:STDOUT:   %Less.ref.loc19: %T.binding.as_type.as.OrderedWith.impl.Less.type.987d0b.1 = name_ref Less, %.loc19_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.e8fdd7.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound: <bound method> = bound_method %int_1.loc19, %Less.ref.loc19 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.66e1cb.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Less.ref.loc19, @T.binding.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.06ee45.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn [concrete = constants.%bound_method.8f808d.2]
// CHECK:STDOUT:   %impl.elem0.loc19_3: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc20: %.184 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem3.loc20 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem3.loc20, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.2c548f.1]
// CHECK:STDOUT:   %.loc20_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1 = specific_constant imports.%Core.GreaterOrEquivalent.b91, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc20: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.fc9354.1 = name_ref GreaterOrEquivalent, %.loc20_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.755c9d.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound: <bound method> = bound_method %int_1.loc20, %GreaterOrEquivalent.ref.loc20 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.b10f47.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn: <specific function> = specific_function %GreaterOrEquivalent.ref.loc20, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ca5627.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn [concrete = constants.%bound_method.2c548f.2]
// CHECK:STDOUT:   %impl.elem0.loc20: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc21: %.5f9 = impl_witness_access constants.%OrderedWith.impl_witness.aae, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem1.loc21 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.caef99.1]
// CHECK:STDOUT:   %.loc21_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1 = specific_constant imports.%Core.LessOrEquivalent.792, @T.binding.as_type.as.OrderedWith.impl.895(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc21: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.39171f.1 = name_ref LessOrEquivalent, %.loc21_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.9bb133.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound: <bound method> = bound_method %int_1.loc21, %LessOrEquivalent.ref.loc21 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.0b467c.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn: <specific function> = specific_function %LessOrEquivalent.ref.loc21, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.d52) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2d74e0.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn [concrete = constants.%bound_method.caef99.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.a93 = impl_witness_access constants.%ImplicitAs.impl_witness.82e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a20]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aae]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.092]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc23_10: type = splice_block %i64.loc23 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc23: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc23: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc23: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc23_16.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_16.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i64 = call %bound_method.loc23_16.2(%int_1.loc23) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc23_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc23_16.2: %i64 = converted %int_1.loc23, %.loc23_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc23_16.2
// CHECK:STDOUT:   %b.ref.loc24: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc24: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc24_5: %.144 = impl_witness_access constants.%EqWith.impl_witness.367, element0 [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.2]
// CHECK:STDOUT:   %bound_method.loc24_5.1: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_5
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_5, @Int.as.EqWith.impl.Equal.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.443739.1]
// CHECK:STDOUT:   %bound_method.loc24_5.2: <bound method> = bound_method %b.ref.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_5: %Int.as.EqWith.impl.Equal.type.e5f20b.1 = specific_constant imports.%Core.Equal.ca9, @Int.as.EqWith.impl.42c(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.1]
// CHECK:STDOUT:   %Equal.ref.loc24: %Int.as.EqWith.impl.Equal.type.e5f20b.1 = name_ref Equal, %.loc24_5 [concrete = constants.%Int.as.EqWith.impl.Equal.e49d18.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.bound.loc24: <bound method> = bound_method %b.ref.loc24, %Equal.ref.loc24
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.specific_fn.loc24: <specific function> = specific_function %Equal.ref.loc24, @Int.as.EqWith.impl.Equal.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.Equal.specific_fn.443739.2]
// CHECK:STDOUT:   %bound_method.loc24_5.3: <bound method> = bound_method %b.ref.loc24, %Int.as.EqWith.impl.Equal.specific_fn.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_8: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_8: <bound method> = bound_method %a.ref.loc24, %impl.elem0.loc24_8
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc24: init %i64 = call %bound_method.loc24_8(%a.ref.loc24)
// CHECK:STDOUT:   %.loc24_8.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_8.2: %i64 = converted %a.ref.loc24, %.loc24_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call.loc24: init bool = call %bound_method.loc24_5.3(%b.ref.loc24, %.loc24_8.2)
// CHECK:STDOUT:   %b.ref.loc25: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc25: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc25: %.b78 = impl_witness_access constants.%EqWith.impl_witness.367, element1 [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.2]
// CHECK:STDOUT:   %bound_method.loc25_5.1: <bound method> = bound_method %b.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @Int.as.EqWith.impl.NotEqual.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.044259.1]
// CHECK:STDOUT:   %bound_method.loc25_5.2: <bound method> = bound_method %b.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_5: %Int.as.EqWith.impl.NotEqual.type.9ab064.1 = specific_constant imports.%Core.NotEqual.334, @Int.as.EqWith.impl.42c(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.1]
// CHECK:STDOUT:   %NotEqual.ref.loc25: %Int.as.EqWith.impl.NotEqual.type.9ab064.1 = name_ref NotEqual, %.loc25_5 [concrete = constants.%Int.as.EqWith.impl.NotEqual.af2dce.1]
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.bound.loc25: <bound method> = bound_method %b.ref.loc25, %NotEqual.ref.loc25
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.specific_fn.loc25: <specific function> = specific_function %NotEqual.ref.loc25, @Int.as.EqWith.impl.NotEqual.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.EqWith.impl.NotEqual.specific_fn.044259.2]
// CHECK:STDOUT:   %bound_method.loc25_5.3: <bound method> = bound_method %b.ref.loc25, %Int.as.EqWith.impl.NotEqual.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_8: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc25: init %i64 = call %bound_method.loc25_8(%a.ref.loc25)
// CHECK:STDOUT:   %.loc25_8.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc25
// CHECK:STDOUT:   %.loc25_8.2: %i64 = converted %a.ref.loc25, %.loc25_8.1
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call.loc25: init bool = call %bound_method.loc25_5.3(%b.ref.loc25, %.loc25_8.2)
// CHECK:STDOUT:   %b.ref.loc26: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc26: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc26: %.8f1 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element2 [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.2]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %b.ref.loc26, %impl.elem2.loc26
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem2.loc26, @Int.as.OrderedWith.impl.Greater.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.1]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %b.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_5: %Int.as.OrderedWith.impl.Greater.type.e91ea2.1 = specific_constant imports.%Core.Greater.cd9, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.1]
// CHECK:STDOUT:   %Greater.ref.loc26: %Int.as.OrderedWith.impl.Greater.type.e91ea2.1 = name_ref Greater, %.loc26_5 [concrete = constants.%Int.as.OrderedWith.impl.Greater.2b7c26.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.bound.loc26: <bound method> = bound_method %b.ref.loc26, %Greater.ref.loc26
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.specific_fn.loc26: <specific function> = specific_function %Greater.ref.loc26, @Int.as.OrderedWith.impl.Greater.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Greater.specific_fn.c215f8.2]
// CHECK:STDOUT:   %bound_method.loc26_5.3: <bound method> = bound_method %b.ref.loc26, %Int.as.OrderedWith.impl.Greater.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_7: <bound method> = bound_method %a.ref.loc26, %impl.elem0.loc26
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc26: init %i64 = call %bound_method.loc26_7(%a.ref.loc26)
// CHECK:STDOUT:   %.loc26_7.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc26
// CHECK:STDOUT:   %.loc26_7.2: %i64 = converted %a.ref.loc26, %.loc26_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call.loc26: init bool = call %bound_method.loc26_5.3(%b.ref.loc26, %.loc26_7.2)
// CHECK:STDOUT:   %b.ref.loc27: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc27: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc27_5: %.6ba = impl_witness_access constants.%OrderedWith.impl_witness.27d, element0 [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.2]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_5
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27_5, @Int.as.OrderedWith.impl.Less.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.e635f0.1]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %b.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_5: %Int.as.OrderedWith.impl.Less.type.0fa05d.1 = specific_constant imports.%Core.Less.df4, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.1]
// CHECK:STDOUT:   %Less.ref.loc27: %Int.as.OrderedWith.impl.Less.type.0fa05d.1 = name_ref Less, %.loc27_5 [concrete = constants.%Int.as.OrderedWith.impl.Less.f7d683.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.bound.loc27: <bound method> = bound_method %b.ref.loc27, %Less.ref.loc27
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.loc27: <specific function> = specific_function %Less.ref.loc27, @Int.as.OrderedWith.impl.Less.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.Less.specific_fn.e635f0.2]
// CHECK:STDOUT:   %bound_method.loc27_5.3: <bound method> = bound_method %b.ref.loc27, %Int.as.OrderedWith.impl.Less.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_7: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_7: <bound method> = bound_method %a.ref.loc27, %impl.elem0.loc27_7
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc27: init %i64 = call %bound_method.loc27_7(%a.ref.loc27)
// CHECK:STDOUT:   %.loc27_7.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc27
// CHECK:STDOUT:   %.loc27_7.2: %i64 = converted %a.ref.loc27, %.loc27_7.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call.loc27: init bool = call %bound_method.loc27_5.3(%b.ref.loc27, %.loc27_7.2)
// CHECK:STDOUT:   %b.ref.loc28: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc28: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc28: %.cf1 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element3 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.2]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %b.ref.loc28, %impl.elem3.loc28
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem3.loc28, @Int.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.1]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %b.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_5: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1 = specific_constant imports.%Core.GreaterOrEquivalent.757, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc28: %Int.as.OrderedWith.impl.GreaterOrEquivalent.type.0e0c4f.1 = name_ref GreaterOrEquivalent, %.loc28_5 [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.520fbe.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc28: <bound method> = bound_method %b.ref.loc28, %GreaterOrEquivalent.ref.loc28
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28: <specific function> = specific_function %GreaterOrEquivalent.ref.loc28, @Int.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.751116.2]
// CHECK:STDOUT:   %bound_method.loc28_5.3: <bound method> = bound_method %b.ref.loc28, %Int.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_8: <bound method> = bound_method %a.ref.loc28, %impl.elem0.loc28
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc28: init %i64 = call %bound_method.loc28_8(%a.ref.loc28)
// CHECK:STDOUT:   %.loc28_8.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc28
// CHECK:STDOUT:   %.loc28_8.2: %i64 = converted %a.ref.loc28, %.loc28_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.GreaterOrEquivalent.call.loc28: init bool = call %bound_method.loc28_5.3(%b.ref.loc28, %.loc28_8.2)
// CHECK:STDOUT:   %b.ref.loc29: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc29: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc29: %.ec7 = impl_witness_access constants.%OrderedWith.impl_witness.27d, element1 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.2]
// CHECK:STDOUT:   %bound_method.loc29_5.1: <bound method> = bound_method %b.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @Int.as.OrderedWith.impl.LessOrEquivalent.2(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.1]
// CHECK:STDOUT:   %bound_method.loc29_5.2: <bound method> = bound_method %b.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_5: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1 = specific_constant imports.%Core.LessOrEquivalent.c5f, @Int.as.OrderedWith.impl.aba(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc29: %Int.as.OrderedWith.impl.LessOrEquivalent.type.f1c847.1 = name_ref LessOrEquivalent, %.loc29_5 [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.56c691.1]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.bound.loc29: <bound method> = bound_method %b.ref.loc29, %LessOrEquivalent.ref.loc29
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29: <specific function> = specific_function %LessOrEquivalent.ref.loc29, @Int.as.OrderedWith.impl.LessOrEquivalent.3(constants.%int_64, constants.%ImplicitAs.facet.660) [concrete = constants.%Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.100dc7.2]
// CHECK:STDOUT:   %bound_method.loc29_5.3: <bound method> = bound_method %b.ref.loc29, %Int.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29: %.912 = impl_witness_access constants.%ImplicitAs.impl_witness.a63, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_8: <bound method> = bound_method %a.ref.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc29: init %i64 = call %bound_method.loc29_8(%a.ref.loc29)
// CHECK:STDOUT:   %.loc29_8.1: %i64 = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29_8.2: %i64 = converted %a.ref.loc29, %.loc29_8.1
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call.loc29: init bool = call %bound_method.loc29_5.3(%b.ref.loc29, %.loc29_8.2)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.c4b: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.unsigned_long_long) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.584: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.c65: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4384: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.195: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.4384) [concrete]
// CHECK:STDOUT:   %.dd7: type = fn_type_with_self_type %ImplicitAs.Convert.type.c4b, %ImplicitAs.facet.195 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bcb: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f87: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bcb = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f87 [concrete]
// CHECK:STDOUT:   %int_1.79a: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.633: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d19: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.88c6c2.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.887181.1: %PassULongLong__carbon_thunk.type.88c6c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.905: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.182 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fb4: %ImplicitAs.type.584 = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.905) [concrete]
// CHECK:STDOUT:   %.e70: type = fn_type_with_self_type %ImplicitAs.Convert.type.c65, %ImplicitAs.facet.fb4 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.03b: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.d3c: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.03b = struct_value () [concrete]
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
// CHECK:STDOUT:   %As.Convert.type.2e6: type = fn_type @As.Convert, @As(%Cpp.unsigned_long_long) [concrete]
// CHECK:STDOUT:   %As.impl_witness.d1f: <witness> = impl_witness imports.%As.impl_witness_table.002 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.45a = facet_value Core.IntLiteral, (%As.impl_witness.d1f) [concrete]
// CHECK:STDOUT:   %.097: type = fn_type_with_self_type %As.Convert.type.2e6, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7c3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d12 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.29a: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.7c3) [concrete]
// CHECK:STDOUT:   %.98c: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.29a [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.109: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.ffb: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.109 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.79a, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.ffb [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc24 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.506: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.03b = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.d3c]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.182 = impl_witness_table (%Core.import_ref.506), @Cpp.unsigned_long_long.as.ImplicitAs.impl.7c6 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.5ac: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type.109 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.ffb]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.dd7 = impl_witness_access constants.%ImplicitAs.impl_witness.4384, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f87]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc15_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %.loc15_56.2: %Cpp.unsigned_long_long = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = value_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.633 = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_60: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc16: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_100.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.d19 = addr_of %.loc16_100.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_unsigned_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_100.2: init %ULongLongResult to %.loc16_100.1 = in_place_init %PassULongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_41: type = splice_block %ULongLongResult.ref.loc16 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc16: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_100.3: ref %ULongLongResult = temporary %.loc16_100.1, %.loc16_100.2
// CHECK:STDOUT:   %.loc16_100.4: %ULongLongResult = acquire_value %.loc16_100.3
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = value_binding cpp_unsigned_long_long_result, %.loc16_100.4
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
// CHECK:STDOUT:   %Cpp.ref.loc19_59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc19_98.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.d19 = addr_of %.loc19_98.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.1(%cpp_compat_ulong_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_98.2: init %ULongLongResult to %.loc19_98.1 = in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_40: type = splice_block %ULongLongResult.ref.loc19 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc19: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_98.3: ref %ULongLongResult = temporary %.loc19_98.1, %.loc19_98.2
// CHECK:STDOUT:   %.loc19_98.4: %ULongLongResult = acquire_value %.loc19_98.3
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = value_binding cpp_compat_ulong_long_result, %.loc19_98.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.157 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc21: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc21_19: type = splice_block %u64 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.e70 = impl_witness_access constants.%ImplicitAs.impl_witness.905, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.d3c]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %cpp_unsigned_long_long.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21(%cpp_unsigned_long_long.ref.loc21)
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call.loc21
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %cpp_unsigned_long_long.ref.loc21, %.loc21_25.1
// CHECK:STDOUT:   %carbon_u64: %u64 = value_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.6f2 = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_72.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.f5e = addr_of %.loc22_72.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.f0c073.2(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_72.2: init %ULongResult to %.loc22_72.1 = in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_29: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_72.3: ref %ULongResult = temporary %.loc22_72.1, %.loc22_72.2
// CHECK:STDOUT:   %.loc22_72.4: %ULongResult = acquire_value %.loc22_72.3
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = value_binding carbon_u64_result, %.loc22_72.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc24_57.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_57: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc24_57.1: <bound method> = bound_method %float, %impl.elem0.loc24_57 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_57, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_57.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_57.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_57.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_57.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_57.4: init %f32.97e to %.loc24_57.3 = initialize_from %.loc24_57.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_57.5: init %array_type to %a.var = array_init (%.loc24_57.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_57.1, %.loc24_57.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_48: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc24: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_23.1: %.097 = impl_witness_access constants.%As.impl_witness.d1f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_23.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_23.1(%int_1.loc24) [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_23.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %.loc24_23.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_23.1 [concrete = constants.%int_1.79a]
// CHECK:STDOUT:     %impl.elem0.loc24_23.2: %.98c = impl_witness_access constants.%ImplicitAs.impl_witness.7c3, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.ffb]
// CHECK:STDOUT:     %bound_method.loc24_23.2: <bound method> = bound_method %.loc24_23.2, %impl.elem0.loc24_23.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call.loc24: init Core.IntLiteral = call %bound_method.loc24_23.2(%.loc24_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.4: Core.IntLiteral = converted %.loc24_23.2, %.loc24_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_72.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound(%.loc22_72.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_98.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc19(%.loc19_98.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_100.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc16(%.loc16_100.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc24(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc22(%self.param: %ULongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc19(%self.param: %ULongLongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.407: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.c4b: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.unsigned_long_long) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4384: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.4384) [concrete]
// CHECK:STDOUT:   %.dd7: type = fn_type_with_self_type %ImplicitAs.Convert.type.c4b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.79a: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.95c: <witness> = impl_witness imports.%Copy.impl_witness_table.759 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.95c) [concrete]
// CHECK:STDOUT:   %.429: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dd7 = impl_witness_access constants.%ImplicitAs.impl_witness.4384, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %.loc9_35: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.429 = impl_witness_access constants.%Copy.impl_witness.95c, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_35, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc9(%.loc9_35)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_13: type = splice_block %unsigned_long_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc9: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %b.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.unsigned_long_long) = "no_op";
// CHECK:STDOUT:
