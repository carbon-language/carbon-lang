// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/arithmetic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/arithmetic.carbon

// --- supported_types.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_signed_char : Cpp.signed_char = 1 as i8;
  let unused carbon_signed_char: i8 = cpp_signed_char;

  let cpp_short : Cpp.short = 1 as i16;
  let unused carbon_short: i16 = cpp_short;

  let cpp_int : Cpp.int = 1 as i32;
  let unused carbon_int: i32 = cpp_int;

  let cpp_unsigned_char : Cpp.unsigned_char = 1 as u8;
  let unused carbon_unsigned_char: u8 = cpp_unsigned_char;

  let cpp_unsigned_short : Cpp.unsigned_short = 1 as u16;
  let unused carbon_unsigned_short: u16 = cpp_unsigned_short;

  let cpp_unsigned_int : Cpp.unsigned_int = 1 as u32;
  let unused carbon_unsigned_int: u32 = cpp_unsigned_int;

  let cpp_float : Cpp.float = 1.0 as f32;
  let unused carbon_float: f32 = cpp_float;

  let cpp_double : Cpp.double = 1.0 as f64;
  let unused carbon_double: f64 = cpp_double;
  //@dump-sem-ir-end
}

// --- fail_todo_unsupported_types.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+7]]:25: error: semantics TODO: `Unsupported: builtin type: long double` [SemanticsTodo]
  // CHECK:STDERR:   let cpp_long_double : Cpp.long_double = 1.0 as f64;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+4]]:25: note: in `Cpp` name lookup for `long_double` [InCppNameLookup]
  // CHECK:STDERR:   let cpp_long_double : Cpp.long_double = 1.0 as f64;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let cpp_long_double : Cpp.long_double = 1.0 as f64;
  let unused carbon_long_double: f64 = cpp_long_double;
  //@dump-sem-ir-end
}

// --- fail_not_builtin.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  // CHECK:STDERR: fail_not_builtin.carbon:[[@LINE+4]]:28: error: member name `not_builtin` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused not_builtin : Cpp.not_builtin = 1;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused not_builtin : Cpp.not_builtin = 1;
}

// --- fail_lookup_in_other_scopes.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
namespace MyNamespace {}
''';

fn F() {
  // CHECK:STDERR: fail_lookup_in_other_scopes.carbon:[[@LINE+4]]:21: error: member name `long` not found in `Cpp.MyNamespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused long : Cpp.MyNamespace.long = 1;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused long : Cpp.MyNamespace.long = 1;
}

// --- override_builtin.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct unsigned_int {
  auto foo() -> unsigned int;
};
''';

fn F() {
  //@dump-sem-ir-begin
  var unsigned_int : Cpp.unsigned_int;
  let unused x: u32 = unsigned_int.foo();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- supported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.696: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.707: type = facet_type <@As, @As(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8f9: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.b39(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.358: %Core.IntLiteral.as.As.impl.Convert.type.8f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab4: <witness> = impl_witness imports.%As.impl_witness_table.268, @Core.IntLiteral.as.As.impl.b39(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.306: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.b39(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a3e: %Core.IntLiteral.as.As.impl.Convert.type.306 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.4fc: %As.type.707 = facet_value Core.IntLiteral, (%As.impl_witness.ab4) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.1cb: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i8, %As.facet.4fc) [concrete]
// CHECK:STDOUT:   %.082: type = fn_type_with_self_type %As.WithSelf.Convert.type.1cb, %As.facet.4fc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.611: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a3e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.cb2: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a3e, @Core.IntLiteral.as.As.impl.Convert.1(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.d88: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.cb2 [concrete]
// CHECK:STDOUT:   %int_1.b49: %i8 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.0e3: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %As.type.e91: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.6fa: <witness> = impl_witness imports.%As.impl_witness_table.268, @Core.IntLiteral.as.As.impl.b39(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b1: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.b39(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.97a: %Core.IntLiteral.as.As.impl.Convert.type.2b1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.86e: %As.type.e91 = facet_value Core.IntLiteral, (%As.impl_witness.6fa) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.0a8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet.86e) [concrete]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %As.WithSelf.Convert.type.0a8, %As.facet.86e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.e04: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.97a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.57d: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.97a, @Core.IntLiteral.as.As.impl.Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.5e0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.57d [concrete]
// CHECK:STDOUT:   %int_1.ab4: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %As.type.3d4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.63d: <witness> = impl_witness imports.%As.impl_witness_table.268, @Core.IntLiteral.as.As.impl.b39(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.9b7: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.b39(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.7f3: %Core.IntLiteral.as.As.impl.Convert.type.9b7 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.344: %As.type.3d4 = facet_value Core.IntLiteral, (%As.impl_witness.63d) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f03: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet.344) [concrete]
// CHECK:STDOUT:   %.b8f: type = fn_type_with_self_type %As.WithSelf.Convert.type.f03, %As.facet.344 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.6f1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.7f3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.187: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.7f3, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.130f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.187 [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.627: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %As.type.a66: type = facet_type <@As, @As(%u8)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.850: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.f96(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3c5: %Core.IntLiteral.as.As.impl.Convert.type.850 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0d2: <witness> = impl_witness imports.%As.impl_witness_table.f68, @Core.IntLiteral.as.As.impl.f96(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.32d: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.f96(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.d6d: %Core.IntLiteral.as.As.impl.Convert.type.32d = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c5a: %As.type.a66 = facet_value Core.IntLiteral, (%As.impl_witness.0d2) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a9c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u8, %As.facet.c5a) [concrete]
// CHECK:STDOUT:   %.b5b: type = fn_type_with_self_type %As.WithSelf.Convert.type.a9c, %As.facet.c5a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.e17: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.d6d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.a1f: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.d6d, @Core.IntLiteral.as.As.impl.Convert.2(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.58e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.a1f [concrete]
// CHECK:STDOUT:   %int_1.77d: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.a81: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %As.type.1a7: type = facet_type <@As, @As(%u16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.327: <witness> = impl_witness imports.%As.impl_witness_table.f68, @Core.IntLiteral.as.As.impl.f96(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.1ab: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.f96(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.778: %Core.IntLiteral.as.As.impl.Convert.type.1ab = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.00a: %As.type.1a7 = facet_value Core.IntLiteral, (%As.impl_witness.327) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.532: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u16, %As.facet.00a) [concrete]
// CHECK:STDOUT:   %.aca: type = fn_type_with_self_type %As.WithSelf.Convert.type.532, %As.facet.00a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.11c: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.778 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.405: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.778, @Core.IntLiteral.as.As.impl.Convert.2(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.897: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.405 [concrete]
// CHECK:STDOUT:   %int_1.024: %u16 = int_value 1 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.167: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %As.type.921: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.13b: <witness> = impl_witness imports.%As.impl_witness_table.f68, @Core.IntLiteral.as.As.impl.f96(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c99: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.f96(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e0f: %Core.IntLiteral.as.As.impl.Convert.type.c99 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c13: %As.type.921 = facet_value Core.IntLiteral, (%As.impl_witness.13b) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.590: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u32, %As.facet.c13) [concrete]
// CHECK:STDOUT:   %.2d7: type = fn_type_with_self_type %As.WithSelf.Convert.type.590, %As.facet.c13 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.51b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e0f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.766: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e0f, @Core.IntLiteral.as.As.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.766 [concrete]
// CHECK:STDOUT:   %int_1.b1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %f32.bf6: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.be3: type = pattern_type %f32.bf6 [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.0d6: type = facet_type <@As, @As(%f32.bf6)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.9ef: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a85: %Core.FloatLiteral.as.As.impl.Convert.type.9ef = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ce0: <witness> = impl_witness imports.%As.impl_witness_table.d4a, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.e49: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.c29: %Core.FloatLiteral.as.As.impl.Convert.type.e49 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.5b7: %As.type.0d6 = facet_value Core.FloatLiteral, (%As.impl_witness.ce0) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.256: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.bf6, %As.facet.5b7) [concrete]
// CHECK:STDOUT:   %.d7a: type = fn_type_with_self_type %As.WithSelf.Convert.type.256, %As.facet.5b7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.cd3: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.c29 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.309: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.c29, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.519: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.309 [concrete]
// CHECK:STDOUT:   %float.4eb: %f32.bf6 = float_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.f86: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.195: type = pattern_type %f64.f86 [concrete]
// CHECK:STDOUT:   %float.6daae3.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.00c: type = facet_type <@As, @As(%f64.f86)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.534: <witness> = impl_witness imports.%As.impl_witness_table.d4a, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.f5f: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.8fc: %Core.FloatLiteral.as.As.impl.Convert.type.f5f = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.153: %As.type.00c = facet_value Core.FloatLiteral, (%As.impl_witness.534) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.99a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.f86, %As.facet.153) [concrete]
// CHECK:STDOUT:   %.06e: type = fn_type_with_self_type %As.WithSelf.Convert.type.99a, %As.facet.153 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.177: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.8fc [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.6d9: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.8fc, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.1302: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.6d9 [concrete]
// CHECK:STDOUT:   %float.524: %f64.f86 = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .signed_char = @F.%i8.2
// CHECK:STDOUT:     .short = @F.%i16.2
// CHECK:STDOUT:     .int = @F.%i32.2
// CHECK:STDOUT:     .unsigned_char = @F.%u8.2
// CHECK:STDOUT:     .unsigned_short = @F.%u16.2
// CHECK:STDOUT:     .unsigned_int = @F.%u32.2
// CHECK:STDOUT:     .float = @F.%f32.2
// CHECK:STDOUT:     .double = @F.%f64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a5e: @Core.IntLiteral.as.As.impl.b39.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.8f9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.b39.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.358)]
// CHECK:STDOUT:   %As.impl_witness_table.268 = impl_witness_table (%Core.import_ref.a5e), @Core.IntLiteral.as.As.impl.b39 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c93: @Core.IntLiteral.as.As.impl.f96.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.850) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.f96.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.3c5)]
// CHECK:STDOUT:   %As.impl_witness_table.f68 = impl_witness_table (%Core.import_ref.c93), @Core.IntLiteral.as.As.impl.f96 [concrete]
// CHECK:STDOUT:   %Core.import_ref.afe: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.9ef) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a85)]
// CHECK:STDOUT:   %As.impl_witness_table.d4a = impl_witness_table (%Core.import_ref.afe), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_signed_char.patt: %pattern_type.696 = value_binding_pattern cpp_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i8.loc8: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.082 = impl_witness_access constants.%As.impl_witness.ab4, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a3e]
// CHECK:STDOUT:   %bound_method.loc8_45.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.611]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.cb2]
// CHECK:STDOUT:   %bound_method.loc8_45.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.d88]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i8 = call %bound_method.loc8_45.2(%int_1.loc8) [concrete = constants.%int_1.b49]
// CHECK:STDOUT:   %.loc8_45.1: %i8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.b49]
// CHECK:STDOUT:   %.loc8_45.2: %i8 = converted %int_1.loc8, %.loc8_45.1 [concrete = constants.%int_1.b49]
// CHECK:STDOUT:   %.loc8_28: type = splice_block %signed_char.ref [concrete = constants.%i8] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %signed_char.ref: type = name_ref signed_char, %i8.2 [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char: %i8 = value_binding cpp_signed_char, %.loc8_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_signed_char.patt: %pattern_type.696 = value_binding_pattern carbon_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char.ref: %i8 = name_ref cpp_signed_char, %cpp_signed_char
// CHECK:STDOUT:   %i8.loc9: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %carbon_signed_char: %i8 = value_binding carbon_signed_char, %cpp_signed_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_short.patt: %pattern_type.0e3 = value_binding_pattern cpp_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16.loc11: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc11: %.671 = impl_witness_access constants.%As.impl_witness.6fa, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.97a]
// CHECK:STDOUT:   %bound_method.loc11_33.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.e04]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.57d]
// CHECK:STDOUT:   %bound_method.loc11_33.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.5e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i16 = call %bound_method.loc11_33.2(%int_1.loc11) [concrete = constants.%int_1.ab4]
// CHECK:STDOUT:   %.loc11_33.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.ab4]
// CHECK:STDOUT:   %.loc11_33.2: %i16 = converted %int_1.loc11, %.loc11_33.1 [concrete = constants.%int_1.ab4]
// CHECK:STDOUT:   %.loc11_22: type = splice_block %short.ref [concrete = constants.%i16] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %short.ref: type = name_ref short, %i16.2 [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short: %i16 = value_binding cpp_short, %.loc11_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_short.patt: %pattern_type.0e3 = value_binding_pattern carbon_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short.ref: %i16 = name_ref cpp_short, %cpp_short
// CHECK:STDOUT:   %i16.loc12: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %carbon_short: %i16 = value_binding carbon_short, %cpp_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_int.patt: %pattern_type.17a = value_binding_pattern cpp_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14: %.b8f = impl_witness_access constants.%As.impl_witness.63d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.7f3]
// CHECK:STDOUT:   %bound_method.loc14_29.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.6f1]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.187]
// CHECK:STDOUT:   %bound_method.loc14_29.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14 [concrete = constants.%bound_method.130f]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_29.2(%int_1.loc14) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc14_20: type = splice_block %int.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %int.ref: type = name_ref int, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int: %i32 = value_binding cpp_int, %.loc14_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_int.patt: %pattern_type.17a = value_binding_pattern carbon_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int.ref: %i32 = name_ref cpp_int, %cpp_int
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_int: %i32 = value_binding carbon_int, %cpp_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_char.patt: %pattern_type.627 = value_binding_pattern cpp_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u8.loc17: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %impl.elem0.loc17: %.b5b = impl_witness_access constants.%As.impl_witness.0d2, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.d6d]
// CHECK:STDOUT:   %bound_method.loc17_49.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.e17]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.a1f]
// CHECK:STDOUT:   %bound_method.loc17_49.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.58e]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc17: init %u8 = call %bound_method.loc17_49.2(%int_1.loc17) [concrete = constants.%int_1.77d]
// CHECK:STDOUT:   %.loc17_49.1: %u8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc17 [concrete = constants.%int_1.77d]
// CHECK:STDOUT:   %.loc17_49.2: %u8 = converted %int_1.loc17, %.loc17_49.1 [concrete = constants.%int_1.77d]
// CHECK:STDOUT:   %.loc17_30: type = splice_block %unsigned_char.ref [concrete = constants.%u8] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_char.ref: type = name_ref unsigned_char, %u8.2 [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char: %u8 = value_binding cpp_unsigned_char, %.loc17_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_char.patt: %pattern_type.627 = value_binding_pattern carbon_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char.ref: %u8 = name_ref cpp_unsigned_char, %cpp_unsigned_char
// CHECK:STDOUT:   %u8.loc18: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %carbon_unsigned_char: %u8 = value_binding carbon_unsigned_char, %cpp_unsigned_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_short.patt: %pattern_type.a81 = value_binding_pattern cpp_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u16.loc20: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %impl.elem0.loc20: %.aca = impl_witness_access constants.%As.impl_witness.327, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.778]
// CHECK:STDOUT:   %bound_method.loc20_51.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.11c]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.405]
// CHECK:STDOUT:   %bound_method.loc20_51.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.897]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc20: init %u16 = call %bound_method.loc20_51.2(%int_1.loc20) [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc20_51.1: %u16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc20 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc20_51.2: %u16 = converted %int_1.loc20, %.loc20_51.1 [concrete = constants.%int_1.024]
// CHECK:STDOUT:   %.loc20_31: type = splice_block %unsigned_short.ref [concrete = constants.%u16] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_short.ref: type = name_ref unsigned_short, %u16.2 [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short: %u16 = value_binding cpp_unsigned_short, %.loc20_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_short.patt: %pattern_type.a81 = value_binding_pattern carbon_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short.ref: %u16 = name_ref cpp_unsigned_short, %cpp_unsigned_short
// CHECK:STDOUT:   %u16.loc21: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %carbon_unsigned_short: %u16 = value_binding carbon_unsigned_short, %cpp_unsigned_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_int.patt: %pattern_type.167 = value_binding_pattern cpp_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32.loc23: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc23: %.2d7 = impl_witness_access constants.%As.impl_witness.13b, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e0f]
// CHECK:STDOUT:   %bound_method.loc23_47.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.51b]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.766]
// CHECK:STDOUT:   %bound_method.loc23_47.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc23: init %u32 = call %bound_method.loc23_47.2(%int_1.loc23) [concrete = constants.%int_1.b1d]
// CHECK:STDOUT:   %.loc23_47.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc23 [concrete = constants.%int_1.b1d]
// CHECK:STDOUT:   %.loc23_47.2: %u32 = converted %int_1.loc23, %.loc23_47.1 [concrete = constants.%int_1.b1d]
// CHECK:STDOUT:   %.loc23_29: type = splice_block %unsigned_int.ref [concrete = constants.%u32] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_int.ref: type = name_ref unsigned_int, %u32.2 [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int: %u32 = value_binding cpp_unsigned_int, %.loc23_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_int.patt: %pattern_type.167 = value_binding_pattern carbon_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int.ref: %u32 = name_ref cpp_unsigned_int, %cpp_unsigned_int
// CHECK:STDOUT:   %u32.loc24: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %carbon_unsigned_int: %u32 = value_binding carbon_unsigned_int, %cpp_unsigned_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_float.patt: %pattern_type.be3 = value_binding_pattern cpp_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc26: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %f32.loc26: type = type_literal constants.%f32.bf6 [concrete = constants.%f32.bf6]
// CHECK:STDOUT:   %impl.elem0.loc26: %.d7a = impl_witness_access constants.%As.impl_witness.ce0, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.c29]
// CHECK:STDOUT:   %bound_method.loc26_35.1: <bound method> = bound_method %float.loc26, %impl.elem0.loc26 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.cd3]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.309]
// CHECK:STDOUT:   %bound_method.loc26_35.2: <bound method> = bound_method %float.loc26, %specific_fn.loc26 [concrete = constants.%bound_method.519]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc26: init %f32.bf6 = call %bound_method.loc26_35.2(%float.loc26) [concrete = constants.%float.4eb]
// CHECK:STDOUT:   %.loc26_35.1: %f32.bf6 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc26 [concrete = constants.%float.4eb]
// CHECK:STDOUT:   %.loc26_35.2: %f32.bf6 = converted %float.loc26, %.loc26_35.1 [concrete = constants.%float.4eb]
// CHECK:STDOUT:   %.loc26_22: type = splice_block %float.ref [concrete = constants.%f32.bf6] {
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %float.ref: type = name_ref float, %f32.2 [concrete = constants.%f32.bf6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float: %f32.bf6 = value_binding cpp_float, %.loc26_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_float.patt: %pattern_type.be3 = value_binding_pattern carbon_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float.ref: %f32.bf6 = name_ref cpp_float, %cpp_float
// CHECK:STDOUT:   %f32.loc27: type = type_literal constants.%f32.bf6 [concrete = constants.%f32.bf6]
// CHECK:STDOUT:   %carbon_float: %f32.bf6 = value_binding carbon_float, %cpp_float.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_double.patt: %pattern_type.195 = value_binding_pattern cpp_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc29: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %f64.loc29: type = type_literal constants.%f64.f86 [concrete = constants.%f64.f86]
// CHECK:STDOUT:   %impl.elem0.loc29: %.06e = impl_witness_access constants.%As.impl_witness.534, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.8fc]
// CHECK:STDOUT:   %bound_method.loc29_37.1: <bound method> = bound_method %float.loc29, %impl.elem0.loc29 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.177]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.6d9]
// CHECK:STDOUT:   %bound_method.loc29_37.2: <bound method> = bound_method %float.loc29, %specific_fn.loc29 [concrete = constants.%bound_method.1302]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc29: init %f64.f86 = call %bound_method.loc29_37.2(%float.loc29) [concrete = constants.%float.524]
// CHECK:STDOUT:   %.loc29_37.1: %f64.f86 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc29 [concrete = constants.%float.524]
// CHECK:STDOUT:   %.loc29_37.2: %f64.f86 = converted %float.loc29, %.loc29_37.1 [concrete = constants.%float.524]
// CHECK:STDOUT:   %.loc29_23: type = splice_block %double.ref [concrete = constants.%f64.f86] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %double.ref: type = name_ref double, %f64.2 [concrete = constants.%f64.f86]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double: %f64.f86 = value_binding cpp_double, %.loc29_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_double.patt: %pattern_type.195 = value_binding_pattern carbon_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double.ref: %f64.f86 = name_ref cpp_double, %cpp_double
// CHECK:STDOUT:   %f64.loc30: type = type_literal constants.%f64.f86 [concrete = constants.%f64.f86]
// CHECK:STDOUT:   %carbon_double: %f64.f86 = value_binding carbon_double, %cpp_double.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.f86: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.00c: type = facet_type <@As, @As(%f64.f86)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.9ef: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a85: %Core.FloatLiteral.as.As.impl.Convert.type.9ef = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.534: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.f5f: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.8fc: %Core.FloatLiteral.as.As.impl.Convert.type.f5f = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.00c = facet_value Core.FloatLiteral, (%As.impl_witness.534) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.99a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.f86, %As.facet) [concrete]
// CHECK:STDOUT:   %.06e: type = fn_type_with_self_type %As.WithSelf.Convert.type.99a, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.8fc [concrete]
// CHECK:STDOUT:   %pattern_type.195: type = pattern_type %f64.f86 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.8fc, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.524: %f64.f86 = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.afe: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.9ef) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a85)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.afe), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_double.patt: <error> = value_binding_pattern cpp_long_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %f64.loc15: type = type_literal constants.%f64.f86 [concrete = constants.%f64.f86]
// CHECK:STDOUT:   %impl.elem0: %.06e = impl_witness_access constants.%As.impl_witness.534, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.8fc]
// CHECK:STDOUT:   %bound_method.loc15_47.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_47.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.f86 = call %bound_method.loc15_47.2(%float) [concrete = constants.%float.524]
// CHECK:STDOUT:   %.loc15_47.1: %f64.f86 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.524]
// CHECK:STDOUT:   %.loc15_47.2: %f64.f86 = converted %float, %.loc15_47.1 [concrete = constants.%float.524]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_long_double: <error> = value_binding cpp_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_double.patt: %pattern_type.195 = value_binding_pattern carbon_long_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_double.ref: <error> = name_ref cpp_long_double, %cpp_long_double [concrete = <error>]
// CHECK:STDOUT:   %f64.loc16: type = type_literal constants.%f64.f86 [concrete = constants.%f64.f86]
// CHECK:STDOUT:   %carbon_long_double: %f64.f86 = value_binding carbon_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- override_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %unsigned_int: type = class_type @unsigned_int [concrete]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %unsigned_int [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.5e4: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.461: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.5e4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.6ea: %T.as_type.as.DefaultOrUnformed.impl.Op.type.461 = struct_value () [symbolic]
// CHECK:STDOUT:   %unsigned_int.Op.type: type = fn_type @unsigned_int.Op [concrete]
// CHECK:STDOUT:   %unsigned_int.Op: %unsigned_int.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4b4: <witness> = custom_witness (%unsigned_int.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.cc8: %Default.type = facet_value %unsigned_int, (%custom_witness.4b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.ca6: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f33, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.cc8) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %unsigned_int, (%DefaultOrUnformed.impl_witness.ca6) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.167: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.type: type = cpp_overload_set_type @unsigned_int.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.type: type = fn_type @unsigned_int.foo [concrete]
// CHECK:STDOUT:   %unsigned_int.foo: %unsigned_int.foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %unsigned_int.cpp_destructor.type: type = fn_type @unsigned_int.cpp_destructor [concrete]
// CHECK:STDOUT:   %unsigned_int.cpp_destructor: %unsigned_int.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_int = %unsigned_int.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.decl: type = class_decl @unsigned_int [concrete = constants.%unsigned_int] {} {}
// CHECK:STDOUT:   %Core.import_ref.d02: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.461) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.6ea)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f33 = impl_witness_table (%Core.import_ref.d02), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %unsigned_int.foo.decl: %unsigned_int.foo.type = fn_decl @unsigned_int.foo [concrete = constants.%unsigned_int.foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unsigned_int.patt: %pattern_type.bac = ref_binding_pattern unsigned_int [concrete]
// CHECK:STDOUT:     %unsigned_int.var_patt: %pattern_type.bac = var_pattern %unsigned_int.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.var: ref %unsigned_int = var %unsigned_int.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%unsigned_int, (constants.%DefaultOrUnformed.impl_witness.ca6) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc12_38.1: %DefaultOrUnformed.type = converted constants.%unsigned_int, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_38.1 [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   %.loc12_38.2: type = converted %.loc12_38.1, %as_type [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_3: ref %unsigned_int = splice_block %unsigned_int.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %unsigned_int to %.loc12_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %unsigned_int.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc12_25: type = splice_block %unsigned_int.ref.loc12 [concrete = constants.%unsigned_int] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_int.ref.loc12: type = name_ref unsigned_int, imports.%unsigned_int.decl [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int: ref %unsigned_int = ref_binding unsigned_int, %unsigned_int.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.167 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.ref.loc13: ref %unsigned_int = name_ref unsigned_int, %unsigned_int
// CHECK:STDOUT:   %foo.ref: %unsigned_int.foo.cpp_overload_set.type = name_ref foo, imports.%unsigned_int.foo.cpp_overload_set.value [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %unsigned_int.ref.loc13, %foo.ref
// CHECK:STDOUT:   %unsigned_int.foo.call: init %u32 = call imports.%unsigned_int.foo.decl(%unsigned_int.ref.loc13)
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %.loc13_40.1: %u32 = value_of_initializer %unsigned_int.foo.call
// CHECK:STDOUT:   %.loc13_40.2: %u32 = converted %unsigned_int.foo.call, %.loc13_40.1
// CHECK:STDOUT:   %x: %u32 = value_binding x, %.loc13_40.2
// CHECK:STDOUT:   %unsigned_int.cpp_destructor.bound: <bound method> = bound_method %unsigned_int.var, constants.%unsigned_int.cpp_destructor
// CHECK:STDOUT:   %unsigned_int.cpp_destructor.call: init %empty_tuple.type = call %unsigned_int.cpp_destructor.bound(%unsigned_int.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
