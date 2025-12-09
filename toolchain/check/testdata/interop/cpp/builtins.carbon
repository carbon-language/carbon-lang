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
  let carbon_signed_char: i8 = cpp_signed_char;

  let cpp_short : Cpp.short = 1 as i16;
  let carbon_short: i16 = cpp_short;

  let cpp_int : Cpp.int = 1 as i32;
  let carbon_int: i32 = cpp_int;

  let cpp_unsigned_char : Cpp.unsigned_char = 1 as u8;
  let carbon_unsigned_char: u8 = cpp_unsigned_char;

  let cpp_unsigned_short : Cpp.unsigned_short = 1 as u16;
  let carbon_unsigned_short: u16 = cpp_unsigned_short;

  let cpp_unsigned_int : Cpp.unsigned_int = 1 as u32;
  let carbon_unsigned_int: u32 = cpp_unsigned_int;

  let cpp_float : Cpp.float = 1.0 as f32;
  let carbon_float: f32 = cpp_float;

  let cpp_double : Cpp.double = 1.0 as f64;
  let carbon_double: f64 = cpp_double;
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
  let carbon_long_double: f64 = cpp_long_double;
  //@dump-sem-ir-end
}

// --- fail_not_builtin.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  // CHECK:STDERR: fail_not_builtin.carbon:[[@LINE+4]]:21: error: member name `not_builtin` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let not_builtin : Cpp.not_builtin = 1;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let not_builtin : Cpp.not_builtin = 1;
}

// --- fail_lookup_in_other_scopes.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
namespace MyNamespace {}
''';

fn F() {
  // CHECK:STDERR: fail_lookup_in_other_scopes.carbon:[[@LINE+4]]:14: error: member name `long` not found in `Cpp.MyNamespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let long : Cpp.MyNamespace.long = 1;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let long : Cpp.MyNamespace.long = 1;
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
  let x: u32 = unsigned_int.foo();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- supported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.7d7: type = facet_type <@As, @As(%i8)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.cc1: type = fn_type @As.Convert, @As(%i8) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.be4: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ecf(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.b5e: %Core.IntLiteral.as.As.impl.Convert.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.d54: <witness> = impl_witness imports.%As.impl_witness_table.de3, @Core.IntLiteral.as.As.impl.ecf(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e03: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ecf(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.367: %Core.IntLiteral.as.As.impl.Convert.type.e03 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.6ff: %As.type.7d7 = facet_value Core.IntLiteral, (%As.impl_witness.d54) [concrete]
// CHECK:STDOUT:   %.67f: type = fn_type_with_self_type %As.Convert.type.cc1, %As.facet.6ff [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.bf0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.367 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.d43: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.367, @Core.IntLiteral.as.As.impl.Convert.1(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.c8f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.d43 [concrete]
// CHECK:STDOUT:   %int_1.30e: %i8 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %As.type.771: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %As.impl_witness.f60: <witness> = impl_witness imports.%As.impl_witness_table.de3, @Core.IntLiteral.as.As.impl.ecf(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.432: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ecf(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.f1b: %Core.IntLiteral.as.As.impl.Convert.type.432 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.e18: %As.type.771 = facet_value Core.IntLiteral, (%As.impl_witness.f60) [concrete]
// CHECK:STDOUT:   %.4d2: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet.e18 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.83f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.f1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.1ac: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.f1b, @Core.IntLiteral.as.As.impl.Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.c76: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.1ac [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %As.impl_witness.bb6: <witness> = impl_witness imports.%As.impl_witness_table.de3, @Core.IntLiteral.as.As.impl.ecf(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.b71: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.ecf(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.070: %Core.IntLiteral.as.As.impl.Convert.type.b71 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.4e4: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.bb6) [concrete]
// CHECK:STDOUT:   %.2bb: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet.4e4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.b66: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.070 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.821: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.070, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.644: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.821 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %As.type.340: type = facet_type <@As, @As(%u8)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.a56: type = fn_type @As.Convert, @As(%u8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.175: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.dcb(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.0ab: %Core.IntLiteral.as.As.impl.Convert.type.175 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.8ca: <witness> = impl_witness imports.%As.impl_witness_table.ee4, @Core.IntLiteral.as.As.impl.dcb(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.20b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.dcb(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.263: %Core.IntLiteral.as.As.impl.Convert.type.20b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.2f5: %As.type.340 = facet_value Core.IntLiteral, (%As.impl_witness.8ca) [concrete]
// CHECK:STDOUT:   %.89b: type = fn_type_with_self_type %As.Convert.type.a56, %As.facet.2f5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.dd1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.263 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.6e0: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.263, @Core.IntLiteral.as.As.impl.Convert.2(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.f18: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.6e0 [concrete]
// CHECK:STDOUT:   %int_1.e80: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9db: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %As.type.e67: type = facet_type <@As, @As(%u16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.c7c: type = fn_type @As.Convert, @As(%u16) [concrete]
// CHECK:STDOUT:   %As.impl_witness.e9d: <witness> = impl_witness imports.%As.impl_witness_table.ee4, @Core.IntLiteral.as.As.impl.dcb(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.b7c: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.dcb(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c98: %Core.IntLiteral.as.As.impl.Convert.type.b7c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.84a: %As.type.e67 = facet_value Core.IntLiteral, (%As.impl_witness.e9d) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %As.Convert.type.c7c, %As.facet.84a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.1c1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.c98 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.a4b: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.c98, @Core.IntLiteral.as.As.impl.Convert.2(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.43b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.a4b [concrete]
// CHECK:STDOUT:   %int_1.3e8: %u16 = int_value 1 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %As.type.45b: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.b94: type = fn_type @As.Convert, @As(%u32) [concrete]
// CHECK:STDOUT:   %As.impl_witness.9fc: <witness> = impl_witness imports.%As.impl_witness_table.ee4, @Core.IntLiteral.as.As.impl.dcb(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e12: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.dcb(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.f85: %Core.IntLiteral.as.As.impl.Convert.type.e12 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.857: %As.type.45b = facet_value Core.IntLiteral, (%As.impl_witness.9fc) [concrete]
// CHECK:STDOUT:   %.830: type = fn_type_with_self_type %As.Convert.type.b94, %As.facet.857 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.446: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.f85 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.0da: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.f85, @Core.IntLiteral.as.As.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1bf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.0da [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %float.674bbc.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.d27: type = facet_type <@As, @As(%f32.97e)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.f5e: type = fn_type @As.Convert, @As(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.13a: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e62: %Core.FloatLiteral.as.As.impl.Convert.type.13a = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.d8e: <witness> = impl_witness imports.%As.impl_witness_table.059, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.248: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.1b6: %Core.FloatLiteral.as.As.impl.Convert.type.248 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.e7c: %As.type.d27 = facet_value Core.FloatLiteral, (%As.impl_witness.d8e) [concrete]
// CHECK:STDOUT:   %.a1d: type = fn_type_with_self_type %As.Convert.type.f5e, %As.facet.e7c [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.06b: <bound method> = bound_method %float.674bbc.1, %Core.FloatLiteral.as.As.impl.Convert.1b6 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.42c: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.1b6, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2f9: <bound method> = bound_method %float.674bbc.1, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.42c [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.674bbc.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.6a8: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.8fc: type = fn_type @As.Convert, @As(%f64.d77) [concrete]
// CHECK:STDOUT:   %As.impl_witness.794: <witness> = impl_witness imports.%As.impl_witness_table.059, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.8e7: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b9b: %Core.FloatLiteral.as.As.impl.Convert.type.8e7 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.e99: %As.type.6a8 = facet_value Core.FloatLiteral, (%As.impl_witness.794) [concrete]
// CHECK:STDOUT:   %.ded: type = fn_type_with_self_type %As.Convert.type.8fc, %As.facet.e99 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.24b: <bound method> = bound_method %float.674bbc.2, %Core.FloatLiteral.as.As.impl.Convert.b9b [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.f58: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.b9b, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.bb8: <bound method> = bound_method %float.674bbc.2, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.f58 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .signed_char = @F.%i8.1
// CHECK:STDOUT:     .short = @F.%i16.1
// CHECK:STDOUT:     .int = @F.%i32.1
// CHECK:STDOUT:     .unsigned_char = @F.%u8.1
// CHECK:STDOUT:     .unsigned_short = @F.%u16.1
// CHECK:STDOUT:     .unsigned_int = @F.%u32.1
// CHECK:STDOUT:     .float = @F.%f32.1
// CHECK:STDOUT:     .double = @F.%f64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ef2: @Core.IntLiteral.as.As.impl.ecf.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.be4) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.ecf.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.b5e)]
// CHECK:STDOUT:   %As.impl_witness_table.de3 = impl_witness_table (%Core.import_ref.ef2), @Core.IntLiteral.as.As.impl.ecf [concrete]
// CHECK:STDOUT:   %Core.import_ref.963: @Core.IntLiteral.as.As.impl.dcb.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.175) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.dcb.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.0ab)]
// CHECK:STDOUT:   %As.impl_witness_table.ee4 = impl_witness_table (%Core.import_ref.963), @Core.IntLiteral.as.As.impl.dcb [concrete]
// CHECK:STDOUT:   %Core.import_ref.4ea7: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.13a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e62)]
// CHECK:STDOUT:   %As.impl_witness_table.059 = impl_witness_table (%Core.import_ref.4ea7), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_signed_char.patt: %pattern_type.e3f = value_binding_pattern cpp_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %i8.loc8: type = class_type @Int, @Int(constants.%int_8) [concrete = constants.%i8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.67f = impl_witness_access constants.%As.impl_witness.d54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.367]
// CHECK:STDOUT:   %bound_method.loc8_45.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.bf0]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.d43]
// CHECK:STDOUT:   %bound_method.loc8_45.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.c8f]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i8 = call %bound_method.loc8_45.2(%int_1.loc8) [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_45.1: %i8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_45.2: %i8 = converted %int_1.loc8, %.loc8_45.1 [concrete = constants.%int_1.30e]
// CHECK:STDOUT:   %.loc8_28: type = splice_block %signed_char.ref [concrete = constants.%i8] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %signed_char.ref: type = name_ref signed_char, %i8.1 [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char: %i8 = value_binding cpp_signed_char, %.loc8_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_signed_char.patt: %pattern_type.e3f = value_binding_pattern carbon_signed_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char.ref: %i8 = name_ref cpp_signed_char, %cpp_signed_char
// CHECK:STDOUT:   %.loc9: type = splice_block %i8.loc9 [concrete = constants.%i8] {
// CHECK:STDOUT:     %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %i8.loc9: type = class_type @Int, @Int(constants.%int_8) [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_signed_char: %i8 = value_binding carbon_signed_char, %cpp_signed_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_short.patt: %pattern_type.2f8 = value_binding_pattern cpp_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16.loc11: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.loc11: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc11: %.4d2 = impl_witness_access constants.%As.impl_witness.f60, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.f1b]
// CHECK:STDOUT:   %bound_method.loc11_33.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.83f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.1ac]
// CHECK:STDOUT:   %bound_method.loc11_33.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.c76]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i16 = call %bound_method.loc11_33.2(%int_1.loc11) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_33.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_33.2: %i16 = converted %int_1.loc11, %.loc11_33.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc11_22: type = splice_block %short.ref [concrete = constants.%i16] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %short.ref: type = name_ref short, %i16.1 [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short: %i16 = value_binding cpp_short, %.loc11_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_short.patt: %pattern_type.2f8 = value_binding_pattern carbon_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short.ref: %i16 = name_ref cpp_short, %cpp_short
// CHECK:STDOUT:   %.loc12: type = splice_block %i16.loc12 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc12: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc12: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_short: %i16 = value_binding carbon_short, %cpp_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_int.patt: %pattern_type.7ce = value_binding_pattern cpp_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14: %.2bb = impl_witness_access constants.%As.impl_witness.bb6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.070]
// CHECK:STDOUT:   %bound_method.loc14_29.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.b66]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.821]
// CHECK:STDOUT:   %bound_method.loc14_29.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14 [concrete = constants.%bound_method.644]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_29.2(%int_1.loc14) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_20: type = splice_block %int.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %int.ref: type = name_ref int, %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int: %i32 = value_binding cpp_int, %.loc14_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_int.patt: %pattern_type.7ce = value_binding_pattern carbon_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int.ref: %i32 = name_ref cpp_int, %cpp_int
// CHECK:STDOUT:   %.loc15: type = splice_block %i32.loc15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_int: %i32 = value_binding carbon_int, %cpp_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_char.patt: %pattern_type.8f3 = value_binding_pattern cpp_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %u8.loc17: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:   %impl.elem0.loc17: %.89b = impl_witness_access constants.%As.impl_witness.8ca, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.263]
// CHECK:STDOUT:   %bound_method.loc17_49.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.dd1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.6e0]
// CHECK:STDOUT:   %bound_method.loc17_49.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.f18]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc17: init %u8 = call %bound_method.loc17_49.2(%int_1.loc17) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_49.1: %u8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc17 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_49.2: %u8 = converted %int_1.loc17, %.loc17_49.1 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc17_30: type = splice_block %unsigned_char.ref [concrete = constants.%u8] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_char.ref: type = name_ref unsigned_char, %u8.1 [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char: %u8 = value_binding cpp_unsigned_char, %.loc17_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_char.patt: %pattern_type.8f3 = value_binding_pattern carbon_unsigned_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char.ref: %u8 = name_ref cpp_unsigned_char, %cpp_unsigned_char
// CHECK:STDOUT:   %.loc18: type = splice_block %u8.loc18 [concrete = constants.%u8] {
// CHECK:STDOUT:     %int_8.loc18: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %u8.loc18: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_char: %u8 = value_binding carbon_unsigned_char, %cpp_unsigned_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_short.patt: %pattern_type.9db = value_binding_pattern cpp_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16.loc20: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %u16.loc20: type = class_type @UInt, @UInt(constants.%int_16) [concrete = constants.%u16]
// CHECK:STDOUT:   %impl.elem0.loc20: %.2ec = impl_witness_access constants.%As.impl_witness.e9d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.c98]
// CHECK:STDOUT:   %bound_method.loc20_51.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.1c1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.a4b]
// CHECK:STDOUT:   %bound_method.loc20_51.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.43b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc20: init %u16 = call %bound_method.loc20_51.2(%int_1.loc20) [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_51.1: %u16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc20 [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_51.2: %u16 = converted %int_1.loc20, %.loc20_51.1 [concrete = constants.%int_1.3e8]
// CHECK:STDOUT:   %.loc20_31: type = splice_block %unsigned_short.ref [concrete = constants.%u16] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_short.ref: type = name_ref unsigned_short, %u16.1 [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short: %u16 = value_binding cpp_unsigned_short, %.loc20_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_short.patt: %pattern_type.9db = value_binding_pattern carbon_unsigned_short [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short.ref: %u16 = name_ref cpp_unsigned_short, %cpp_unsigned_short
// CHECK:STDOUT:   %.loc21: type = splice_block %u16.loc21 [concrete = constants.%u16] {
// CHECK:STDOUT:     %int_16.loc21: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %u16.loc21: type = class_type @UInt, @UInt(constants.%int_16) [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_short: %u16 = value_binding carbon_unsigned_short, %cpp_unsigned_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_int.patt: %pattern_type.4a9 = value_binding_pattern cpp_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %u32.loc23: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc23: %.830 = impl_witness_access constants.%As.impl_witness.9fc, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.f85]
// CHECK:STDOUT:   %bound_method.loc23_47.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.446]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.0da]
// CHECK:STDOUT:   %bound_method.loc23_47.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.1bf]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc23: init %u32 = call %bound_method.loc23_47.2(%int_1.loc23) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_47.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc23 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_47.2: %u32 = converted %int_1.loc23, %.loc23_47.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc23_29: type = splice_block %unsigned_int.ref [concrete = constants.%u32] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_int.ref: type = name_ref unsigned_int, %u32.1 [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int: %u32 = value_binding cpp_unsigned_int, %.loc23_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_int.patt: %pattern_type.4a9 = value_binding_pattern carbon_unsigned_int [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int.ref: %u32 = name_ref cpp_unsigned_int, %cpp_unsigned_int
// CHECK:STDOUT:   %.loc24: type = splice_block %u32.loc24 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc24: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_int: %u32 = value_binding carbon_unsigned_int, %cpp_unsigned_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_float.patt: %pattern_type.201 = value_binding_pattern cpp_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc26: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674bbc.1]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %f32.loc26: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %impl.elem0.loc26: %.a1d = impl_witness_access constants.%As.impl_witness.d8e, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.1b6]
// CHECK:STDOUT:   %bound_method.loc26_35.1: <bound method> = bound_method %float.loc26, %impl.elem0.loc26 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.06b]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.42c]
// CHECK:STDOUT:   %bound_method.loc26_35.2: <bound method> = bound_method %float.loc26, %specific_fn.loc26 [concrete = constants.%bound_method.2f9]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc26: init %f32.97e = call %bound_method.loc26_35.2(%float.loc26) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_35.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc26 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_35.2: %f32.97e = converted %float.loc26, %.loc26_35.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_22: type = splice_block %float.ref [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %float.ref: type = name_ref float, %f32.1 [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float: %f32.97e = value_binding cpp_float, %.loc26_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_float.patt: %pattern_type.201 = value_binding_pattern carbon_float [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float.ref: %f32.97e = name_ref cpp_float, %cpp_float
// CHECK:STDOUT:   %.loc27: type = splice_block %f32.loc27 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc27: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_float: %f32.97e = value_binding carbon_float, %cpp_float.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_double.patt: %pattern_type.0ae = value_binding_pattern cpp_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc29: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674bbc.2]
// CHECK:STDOUT:   %int_64.loc29: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %f64.loc29: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0.loc29: %.ded = impl_witness_access constants.%As.impl_witness.794, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.b9b]
// CHECK:STDOUT:   %bound_method.loc29_37.1: <bound method> = bound_method %float.loc29, %impl.elem0.loc29 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.24b]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.f58]
// CHECK:STDOUT:   %bound_method.loc29_37.2: <bound method> = bound_method %float.loc29, %specific_fn.loc29 [concrete = constants.%bound_method.bb8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc29: init %f64.d77 = call %bound_method.loc29_37.2(%float.loc29) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_37.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc29 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_37.2: %f64.d77 = converted %float.loc29, %.loc29_37.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc29_23: type = splice_block %double.ref [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %double.ref: type = name_ref double, %f64.1 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double: %f64.d77 = value_binding cpp_double, %.loc29_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_double.patt: %pattern_type.0ae = value_binding_pattern carbon_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double.ref: %f64.d77 = name_ref cpp_double, %cpp_double
// CHECK:STDOUT:   %.loc30: type = splice_block %f64.loc30 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc30: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc30: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_double: %f64.d77 = value_binding carbon_double, %cpp_double.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.6a8: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.8fc: type = fn_type @As.Convert, @As(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.13a: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e62: %Core.FloatLiteral.as.As.impl.Convert.type.13a = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.794: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.8e7: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b9b: %Core.FloatLiteral.as.As.impl.Convert.type.8e7 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.6a8 = facet_value Core.FloatLiteral, (%As.impl_witness.794) [concrete]
// CHECK:STDOUT:   %.ded: type = fn_type_with_self_type %As.Convert.type.8fc, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.As.impl.Convert.b9b [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.b9b, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4ea7: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.13a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e62)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.4ea7), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_double.patt: <error> = value_binding_pattern cpp_long_double [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %f64.loc15: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0: %.ded = impl_witness_access constants.%As.impl_witness.794, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.b9b]
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
// CHECK:STDOUT:   %.loc16: type = splice_block %f64.loc16 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc16: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %unsigned_int.foo.decl: %unsigned_int.foo.type = fn_decl @unsigned_int.foo [concrete = constants.%unsigned_int.foo] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5ec = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5ec = ref_param_pattern %self.patt, call_param0 [concrete]
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
// CHECK:STDOUT:   %.loc12: type = splice_block %unsigned_int.ref.loc12 [concrete = constants.%unsigned_int] {
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
// CHECK:STDOUT:   %.loc13_10: type = splice_block %u32 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_33.1: %u32 = value_of_initializer %unsigned_int.foo.call
// CHECK:STDOUT:   %.loc13_33.2: %u32 = converted %unsigned_int.foo.call, %.loc13_33.1
// CHECK:STDOUT:   %x: %u32 = value_binding x, %.loc13_33.2
// CHECK:STDOUT:   %unsigned_int.cpp_destructor.bound: <bound method> = bound_method %unsigned_int.var, constants.%unsigned_int.cpp_destructor
// CHECK:STDOUT:   %unsigned_int.cpp_destructor.call: init %empty_tuple.type = call %unsigned_int.cpp_destructor.bound(%unsigned_int.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
