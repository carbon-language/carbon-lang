// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/builtins.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/builtins.carbon

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
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.b4c: type = facet_type <@As, @As(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ae3(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0bb: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl.ae3(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.483: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ae3(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bf2: %Core.IntLiteral.as.As.impl.Convert.type.483 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.6fc: %As.type.b4c = facet_value Core.IntLiteral, (%As.impl_witness.0bb) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a98: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i8, %As.facet.6fc) [concrete]
// CHECK:STDOUT:   %.57e: type = fn_type_with_self_type %As.WithSelf.Convert.type.a98, %As.facet.6fc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.bf2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.d98: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.bf2, @Core.IntLiteral.as.As.impl.Convert.1(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.1fe: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.d98 [concrete]
// CHECK:STDOUT:   %int_1.30e: %i8 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.b61: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl.ae3(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c60: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ae3(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a42: %Core.IntLiteral.as.As.impl.Convert.type.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.071: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet.071) [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %As.WithSelf.Convert.type.2c0, %As.facet.071 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.896: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.00b: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a42, @Core.IntLiteral.as.As.impl.Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.4da: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.00b [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl.ae3(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ae3(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.d6e: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet.d6e) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet.d6e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.bd3: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.fab: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.290: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.fab [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %As.type.714: type = facet_type <@As, @As(%u8)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b1: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.5fe(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.979: %Core.IntLiteral.as.As.impl.Convert.type.2b1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ce7: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl.5fe(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.664: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.5fe(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.edb: %Core.IntLiteral.as.As.impl.Convert.type.664 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.5e2: %As.type.714 = facet_value Core.IntLiteral, (%As.impl_witness.ce7) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2ea: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u8, %As.facet.5e2) [concrete]
// CHECK:STDOUT:   %.3d2: type = fn_type_with_self_type %As.WithSelf.Convert.type.2ea, %As.facet.5e2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.f5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.edb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.130: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.edb, @Core.IntLiteral.as.As.impl.Convert.2(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.6b2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.130 [concrete]
// CHECK:STDOUT:   %int_1.e80: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9db: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %As.type.b84: type = facet_type <@As, @As(%u16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.e55: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl.5fe(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c43: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.5fe(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dab: %Core.IntLiteral.as.As.impl.Convert.type.c43 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.810: %As.type.b84 = facet_value Core.IntLiteral, (%As.impl_witness.e55) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.301: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u16, %As.facet.810) [concrete]
// CHECK:STDOUT:   %.f78: type = fn_type_with_self_type %As.WithSelf.Convert.type.301, %As.facet.810 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.e01: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.dab [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.b56: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.dab, @Core.IntLiteral.as.As.impl.Convert.2(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.6f0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.b56 [concrete]
// CHECK:STDOUT:   %int_1.3e8: %u16 = int_value 1 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %As.type.346: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.e1d: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl.5fe(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.3e7: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.5fe(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bbd: %Core.IntLiteral.as.As.impl.Convert.type.3e7 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.560: %As.type.346 = facet_value Core.IntLiteral, (%As.impl_witness.e1d) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.89a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u32, %As.facet.560) [concrete]
// CHECK:STDOUT:   %.1a4: type = fn_type_with_self_type %As.WithSelf.Convert.type.89a, %As.facet.560 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.a65: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.bbd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.f97: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.bbd, @Core.IntLiteral.as.As.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f17: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.f97 [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.9fc: type = facet_type <@As, @As(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.882: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.27d: %Core.FloatLiteral.as.As.impl.Convert.type.882 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.748: <witness> = impl_witness imports.%As.impl_witness_table.7c1, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.847: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.683: %Core.FloatLiteral.as.As.impl.Convert.type.847 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.58f: %As.type.9fc = facet_value Core.FloatLiteral, (%As.impl_witness.748) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.85f: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.97e, %As.facet.58f) [concrete]
// CHECK:STDOUT:   %.49e: type = fn_type_with_self_type %As.WithSelf.Convert.type.85f, %As.facet.58f [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.04c: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.683 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.10a: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.683, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.4ec: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.10a [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.6daae3.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.a57: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.187: <witness> = impl_witness imports.%As.impl_witness_table.7c1, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.07c: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.4b9: %Core.FloatLiteral.as.As.impl.Convert.type.07c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.750: %As.type.a57 = facet_value Core.FloatLiteral, (%As.impl_witness.187) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dab: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.d77, %As.facet.750) [concrete]
// CHECK:STDOUT:   %.293: type = fn_type_with_self_type %As.WithSelf.Convert.type.dab, %As.facet.750 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.9c5: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.4b9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.b2e: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.4b9, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.18b: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.b2e [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.ae3.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.ae3.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl.ae3 [concrete]
// CHECK:STDOUT:   %Core.import_ref.600: @Core.IntLiteral.as.As.impl.5fe.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.2b1) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5fe.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.979)]
// CHECK:STDOUT:   %As.impl_witness_table.7eb = impl_witness_table (%Core.import_ref.600), @Core.IntLiteral.as.As.impl.5fe [concrete]
// CHECK:STDOUT:   %Core.import_ref.509: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.882) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.27d)]
// CHECK:STDOUT:   %As.impl_witness_table.7c1 = impl_witness_table (%Core.import_ref.509), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_signed_char.patt: %pattern_type.e3f = value_binding_pattern cpp_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i8.loc8: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.57e = impl_witness_access constants.%As.impl_witness.0bb, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bf2]
// CHECK:STDOUT:   %bound_method.loc8_45.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.c11]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.d98]
// CHECK:STDOUT:   %bound_method.loc8_45.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.1fe]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i8 = call %bound_method.loc8_45.2(%int_1.loc8) [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_45.1: %i8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_45.2: %i8 = converted %int_1.loc8, %.loc8_45.1 [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_28: type = splice_block %signed_char.ref [concrete = constants.%i8] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %signed_char.ref: type = name_ref signed_char, %i8.2 [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char: %i8 = value_binding cpp_signed_char, %.loc8_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_signed_char.patt: %pattern_type.e3f = value_binding_pattern carbon_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char.ref: %i8 = name_ref cpp_signed_char, %cpp_signed_char
// CHECK:STDOUT:   %i8.loc9: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %carbon_signed_char: %i8 = value_binding carbon_signed_char, %cpp_signed_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_short.patt: %pattern_type.2f8 = value_binding_pattern cpp_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16.loc11: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc11: %.70c = impl_witness_access constants.%As.impl_witness.b61, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a42]
// CHECK:STDOUT:   %bound_method.loc11_33.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.896]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.00b]
// CHECK:STDOUT:   %bound_method.loc11_33.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.4da]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i16 = call %bound_method.loc11_33.2(%int_1.loc11) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_33.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_33.2: %i16 = converted %int_1.loc11, %.loc11_33.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_22: type = splice_block %short.ref [concrete = constants.%i16] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %short.ref: type = name_ref short, %i16.2 [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short: %i16 = value_binding cpp_short, %.loc11_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_short.patt: %pattern_type.2f8 = value_binding_pattern carbon_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short.ref: %i16 = name_ref cpp_short, %cpp_short
// CHECK:STDOUT:   %i16.loc12: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %carbon_short: %i16 = value_binding carbon_short, %cpp_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_int.patt: %pattern_type.7ce = value_binding_pattern cpp_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc14_29.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.bd3]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.fab]
// CHECK:STDOUT:   %bound_method.loc14_29.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14 [concrete = constants.%bound_method.290]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_29.2(%int_1.loc14) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_20: type = splice_block %int.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %int.ref: type = name_ref int, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int: %i32 = value_binding cpp_int, %.loc14_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_int.patt: %pattern_type.7ce = value_binding_pattern carbon_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int.ref: %i32 = name_ref cpp_int, %cpp_int
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_int: %i32 = value_binding carbon_int, %cpp_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_char.patt: %pattern_type.8f3 = value_binding_pattern cpp_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u8.loc17: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %impl.elem0.loc17: %.3d2 = impl_witness_access constants.%As.impl_witness.ce7, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.edb]
// CHECK:STDOUT:   %bound_method.loc17_49.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.f5f]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.130]
// CHECK:STDOUT:   %bound_method.loc17_49.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.6b2]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc17: init %u8 = call %bound_method.loc17_49.2(%int_1.loc17) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_49.1: %u8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc17 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_49.2: %u8 = converted %int_1.loc17, %.loc17_49.1 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_30: type = splice_block %unsigned_char.ref [concrete = constants.%u8] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_char.ref: type = name_ref unsigned_char, %u8.2 [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char: %u8 = value_binding cpp_unsigned_char, %.loc17_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_char.patt: %pattern_type.8f3 = value_binding_pattern carbon_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char.ref: %u8 = name_ref cpp_unsigned_char, %cpp_unsigned_char
// CHECK:STDOUT:   %u8.loc18: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %carbon_unsigned_char: %u8 = value_binding carbon_unsigned_char, %cpp_unsigned_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_short.patt: %pattern_type.9db = value_binding_pattern cpp_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u16.loc20: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %impl.elem0.loc20: %.f78 = impl_witness_access constants.%As.impl_witness.e55, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.dab]
// CHECK:STDOUT:   %bound_method.loc20_51.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.e01]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.b56]
// CHECK:STDOUT:   %bound_method.loc20_51.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.6f0]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc20: init %u16 = call %bound_method.loc20_51.2(%int_1.loc20) [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_51.1: %u16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc20 [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_51.2: %u16 = converted %int_1.loc20, %.loc20_51.1 [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_31: type = splice_block %unsigned_short.ref [concrete = constants.%u16] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_short.ref: type = name_ref unsigned_short, %u16.2 [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short: %u16 = value_binding cpp_unsigned_short, %.loc20_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_short.patt: %pattern_type.9db = value_binding_pattern carbon_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short.ref: %u16 = name_ref cpp_unsigned_short, %cpp_unsigned_short
// CHECK:STDOUT:   %u16.loc21: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %carbon_unsigned_short: %u16 = value_binding carbon_unsigned_short, %cpp_unsigned_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_int.patt: %pattern_type.4a9 = value_binding_pattern cpp_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32.loc23: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc23: %.1a4 = impl_witness_access constants.%As.impl_witness.e1d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bbd]
// CHECK:STDOUT:   %bound_method.loc23_47.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.a65]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.f97]
// CHECK:STDOUT:   %bound_method.loc23_47.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.f17]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc23: init %u32 = call %bound_method.loc23_47.2(%int_1.loc23) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_47.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc23 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_47.2: %u32 = converted %int_1.loc23, %.loc23_47.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_29: type = splice_block %unsigned_int.ref [concrete = constants.%u32] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_int.ref: type = name_ref unsigned_int, %u32.2 [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int: %u32 = value_binding cpp_unsigned_int, %.loc23_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_int.patt: %pattern_type.4a9 = value_binding_pattern carbon_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int.ref: %u32 = name_ref cpp_unsigned_int, %cpp_unsigned_int
// CHECK:STDOUT:   %u32.loc24: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %carbon_unsigned_int: %u32 = value_binding carbon_unsigned_int, %cpp_unsigned_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_float.patt: %pattern_type.201 = value_binding_pattern cpp_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc26: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %f32.loc26: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %impl.elem0.loc26: %.49e = impl_witness_access constants.%As.impl_witness.748, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.683]
// CHECK:STDOUT:   %bound_method.loc26_35.1: <bound method> = bound_method %float.loc26, %impl.elem0.loc26 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.04c]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.10a]
// CHECK:STDOUT:   %bound_method.loc26_35.2: <bound method> = bound_method %float.loc26, %specific_fn.loc26 [concrete = constants.%bound_method.4ec]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc26: init %f32.97e = call %bound_method.loc26_35.2(%float.loc26) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_35.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc26 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_35.2: %f32.97e = converted %float.loc26, %.loc26_35.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_22: type = splice_block %float.ref [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %float.ref: type = name_ref float, %f32.2 [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float: %f32.97e = value_binding cpp_float, %.loc26_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_float.patt: %pattern_type.201 = value_binding_pattern carbon_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float.ref: %f32.97e = name_ref cpp_float, %cpp_float
// CHECK:STDOUT:   %f32.loc27: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %carbon_float: %f32.97e = value_binding carbon_float, %cpp_float.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_double.patt: %pattern_type.0ae = value_binding_pattern cpp_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc29: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %f64.loc29: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0.loc29: %.293 = impl_witness_access constants.%As.impl_witness.187, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.4b9]
// CHECK:STDOUT:   %bound_method.loc29_37.1: <bound method> = bound_method %float.loc29, %impl.elem0.loc29 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.9c5]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.b2e]
// CHECK:STDOUT:   %bound_method.loc29_37.2: <bound method> = bound_method %float.loc29, %specific_fn.loc29 [concrete = constants.%bound_method.18b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc29: init %f64.d77 = call %bound_method.loc29_37.2(%float.loc29) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_37.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc29 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_37.2: %f64.d77 = converted %float.loc29, %.loc29_37.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_23: type = splice_block %double.ref [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %double.ref: type = name_ref double, %f64.2 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double: %f64.d77 = value_binding cpp_double, %.loc29_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_double.patt: %pattern_type.0ae = value_binding_pattern carbon_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double.ref: %f64.d77 = name_ref cpp_double, %cpp_double
// CHECK:STDOUT:   %f64.loc30: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %carbon_double: %f64.d77 = value_binding carbon_double, %cpp_double.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.a57: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.882: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.27d: %Core.FloatLiteral.as.As.impl.Convert.type.882 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.187: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.07c: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.4b9: %Core.FloatLiteral.as.As.impl.Convert.type.07c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a57 = facet_value Core.FloatLiteral, (%As.impl_witness.187) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dab: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.d77, %As.facet) [concrete]
// CHECK:STDOUT:   %.293: type = fn_type_with_self_type %As.WithSelf.Convert.type.dab, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.4b9 [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.4b9, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.509: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.882) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.27d)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.509), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_double.patt: <error> = value_binding_pattern cpp_long_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %f64.loc15: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0: %.293 = impl_witness_access constants.%As.impl_witness.187, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.4b9]
// CHECK:STDOUT:   %bound_method.loc15_47.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_47.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.d77 = call %bound_method.loc15_47.2(%float) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc15_47.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc15_47.2: %f64.d77 = converted %float, %.loc15_47.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_long_double: <error> = value_binding cpp_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_double.patt: %pattern_type.0ae = value_binding_pattern carbon_long_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_double.ref: <error> = name_ref cpp_long_double, %cpp_long_double [concrete = <error>]
// CHECK:STDOUT:   %f64.loc16: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %carbon_long_double: %f64.d77 = value_binding carbon_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- override_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %unsigned_int: type = class_type @unsigned_int [concrete]
// CHECK:STDOUT:   %pattern_type.5ec: type = pattern_type %unsigned_int [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.4c4: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%unsigned_int) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %unsigned_int, (%DefaultOrUnformed.impl_witness.4c4) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %unsigned_int.foo.decl: %unsigned_int.foo.type = fn_decl @unsigned_int.foo [concrete = constants.%unsigned_int.foo] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5ec = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5ec = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: ref %unsigned_int = ref_param call_param0
// CHECK:STDOUT:     %self: ref %unsigned_int = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unsigned_int.patt: %pattern_type.5ec = ref_binding_pattern unsigned_int [concrete]
// CHECK:STDOUT:     %unsigned_int.var_patt: %pattern_type.5ec = var_pattern %unsigned_int.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.var: ref %unsigned_int = var %unsigned_int.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%unsigned_int, (constants.%DefaultOrUnformed.impl_witness.4c4) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc12_38.1: %DefaultOrUnformed.type = converted constants.%unsigned_int, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_38.1 [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   %.loc12_38.2: type = converted %.loc12_38.1, %as_type [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_3: ref %unsigned_int = splice_block %unsigned_int.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %unsigned_int to %.loc12_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %unsigned_int.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc12_25: type = splice_block %unsigned_int.ref.loc12 [concrete = constants.%unsigned_int] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_int.ref.loc12: type = name_ref unsigned_int, imports.%unsigned_int.decl [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int: ref %unsigned_int = ref_binding unsigned_int, %unsigned_int.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.4a9 = value_binding_pattern x [concrete]
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
