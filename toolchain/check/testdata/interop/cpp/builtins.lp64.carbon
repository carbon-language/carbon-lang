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
// CHECK:STDOUT:   %.loc11_43.4: init %f32.97e = initialize_from %.loc11_43.2 to %.loc11_43.3 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_43.5: init %array_type = array_init (%.loc11_43.4) to %a.var [concrete = constants.%array]
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
// CHECK:STDOUT:   %.loc11_52.4: init %f32.97e = initialize_from %.loc11_52.2 to %.loc11_52.3 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_52.5: init %array_type = array_init (%.loc11_52.4) to %a.var [concrete = constants.%array]
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
// CHECK:STDOUT:   %Core.import_ref.91c: %Cpp.long_long.as.ImplicitAs.impl.Convert.type.194 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.c03]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.563 = impl_witness_table (%Core.import_ref.91c), @Cpp.long_long.as.ImplicitAs.impl.034 [concrete]
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
// CHECK:STDOUT:   %.loc16_80.2: init %LongLongResult = in_place_init %PassLongLong__carbon_thunk.call.loc16, %.loc16_80.1
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
// CHECK:STDOUT:   %.loc19_94.2: init %LongLongResult = in_place_init %PassLongLong__carbon_thunk.call.loc19, %.loc19_94.1
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
// CHECK:STDOUT:   %.loc22_70.2: init %LongResult = in_place_init %PassLongLong__carbon_thunk.call.loc22, %.loc22_70.1
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
// CHECK:STDOUT:   %.loc24_48.4: init %f32.97e = initialize_from %.loc24_48.2 to %.loc24_48.3 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_48.5: init %array_type = array_init (%.loc24_48.4) to %a.var [concrete = constants.%array]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.438: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.517 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.195: %ImplicitAs.type.407 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.438) [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.dd7 = impl_witness_access constants.%ImplicitAs.impl_witness.438, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f87]
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
// CHECK:STDOUT:   %.loc16_100.2: init %ULongLongResult = in_place_init %PassULongLong__carbon_thunk.call.loc16, %.loc16_100.1
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
// CHECK:STDOUT:   %.loc19_98.2: init %ULongLongResult = in_place_init %PassULongLong__carbon_thunk.call.loc19, %.loc19_98.1
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
// CHECK:STDOUT:   %.loc22_72.2: init %ULongResult = in_place_init %PassULongLong__carbon_thunk.call.loc22, %.loc22_72.1
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
// CHECK:STDOUT:   %.loc24_57.4: init %f32.97e = initialize_from %.loc24_57.2 to %.loc24_57.3 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_57.5: init %array_type = array_init (%.loc24_57.4) to %a.var [concrete = constants.%array]
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
