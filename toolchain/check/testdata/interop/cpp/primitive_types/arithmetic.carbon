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
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+3]]:25-39: error: semantics TODO: `Unsupported: builtin type: long double` [SemanticsTodo]
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+2]]:25-39: note: in `Cpp` name lookup for `long_double` [InCppNameLookup]
  // CHECK:STDERR:
  let cpp_long_double : Cpp.long_double = 1.0 as f64;
  let unused carbon_long_double: f64 = cpp_long_double;
  //@dump-sem-ir-end
}

// --- fail_not_builtin.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  // CHECK:STDERR: fail_not_builtin.carbon:[[@LINE+2]]:28-42: error: member name `not_builtin` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  let unused not_builtin : Cpp.not_builtin = 1;
}

// --- fail_lookup_in_other_scopes.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
namespace MyNamespace {}
''';

fn F() {
  // CHECK:STDERR: fail_lookup_in_other_scopes.carbon:[[@LINE+2]]:21-40: error: member name `long` not found in `Cpp.MyNamespace` [MemberNameNotFoundInInstScope]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.060: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %cpp_signed_char.patt: %pattern_type.060 = value_binding_pattern cpp_signed_char [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.88d: type = facet_type <@As, @As(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.753(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.eaa: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl.753(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.d01: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.753(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.af7: %Core.IntLiteral.as.As.impl.Convert.type.d01 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.7d8: %As.type.88d = facet_value Core.IntLiteral, (%As.impl_witness.eaa) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.c54: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i8, %As.facet.7d8) [concrete]
// CHECK:STDOUT:   %.706: type = fn_type_with_self_type %As.WithSelf.Convert.type.c54, %As.facet.7d8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.fa4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.af7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.730: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.af7, @Core.IntLiteral.as.As.impl.Convert.1(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.7a1f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.730 [concrete]
// CHECK:STDOUT:   %int_1.2eb: %i8 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_signed_char.patt: %pattern_type.060 = value_binding_pattern carbon_signed_char [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9bc: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %cpp_short.patt: %pattern_type.9bc = value_binding_pattern cpp_short [concrete]
// CHECK:STDOUT:   %As.type.144: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.4de: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl.753(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.361: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.753(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.262: %Core.IntLiteral.as.As.impl.Convert.type.361 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c1b: %As.type.144 = facet_value Core.IntLiteral, (%As.impl_witness.4de) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.0b5: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet.c1b) [concrete]
// CHECK:STDOUT:   %.f79: type = fn_type_with_self_type %As.WithSelf.Convert.type.0b5, %As.facet.c1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.d94: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.262 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.f24: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.262, @Core.IntLiteral.as.As.impl.Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.1e6: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.f24 [concrete]
// CHECK:STDOUT:   %int_1.6e1: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_short.patt: %pattern_type.9bc = value_binding_pattern carbon_short [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %cpp_int.patt: %pattern_type.d94 = value_binding_pattern cpp_int [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl.753(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.753(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c68: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet.c68) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet.c68 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.022: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.7fd: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.7fd [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_int.patt: %pattern_type.d94 = value_binding_pattern carbon_int [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.e53: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_char.patt: %pattern_type.e53 = value_binding_pattern cpp_unsigned_char [concrete]
// CHECK:STDOUT:   %As.type.05a: type = facet_type <@As, @As(%u8)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.16e: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.d83(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.81c: %Core.IntLiteral.as.As.impl.Convert.type.16e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.a43: <witness> = impl_witness imports.%As.impl_witness_table.43c, @Core.IntLiteral.as.As.impl.d83(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.096: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.d83(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.ff3: %Core.IntLiteral.as.As.impl.Convert.type.096 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.f14: %As.type.05a = facet_value Core.IntLiteral, (%As.impl_witness.a43) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dbd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u8, %As.facet.f14) [concrete]
// CHECK:STDOUT:   %.39f: type = fn_type_with_self_type %As.WithSelf.Convert.type.dbd, %As.facet.f14 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.002: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.ff3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.92f: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.ff3, @Core.IntLiteral.as.As.impl.Convert.2(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.7a1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.92f [concrete]
// CHECK:STDOUT:   %int_1.5dc: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_char.patt: %pattern_type.e53 = value_binding_pattern carbon_unsigned_char [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.bc6: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_short.patt: %pattern_type.bc6 = value_binding_pattern cpp_unsigned_short [concrete]
// CHECK:STDOUT:   %As.type.b6c: type = facet_type <@As, @As(%u16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.8df: <witness> = impl_witness imports.%As.impl_witness_table.43c, @Core.IntLiteral.as.As.impl.d83(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.dc1: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.d83(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.ae8: %Core.IntLiteral.as.As.impl.Convert.type.dc1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.edb: %As.type.b6c = facet_value Core.IntLiteral, (%As.impl_witness.8df) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.14a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u16, %As.facet.edb) [concrete]
// CHECK:STDOUT:   %.5ac: type = fn_type_with_self_type %As.WithSelf.Convert.type.14a, %As.facet.edb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.db4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.ae8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.a7d: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.ae8, @Core.IntLiteral.as.As.impl.Convert.2(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.bd4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.a7d [concrete]
// CHECK:STDOUT:   %int_1.f2e: %u16 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_short.patt: %pattern_type.bc6 = value_binding_pattern carbon_unsigned_short [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %cpp_unsigned_int.patt: %pattern_type.129 = value_binding_pattern cpp_unsigned_int [concrete]
// CHECK:STDOUT:   %As.type.d0a: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.f4f: <witness> = impl_witness imports.%As.impl_witness_table.43c, @Core.IntLiteral.as.As.impl.d83(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ea3: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.d83(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.0bc: %Core.IntLiteral.as.As.impl.Convert.type.ea3 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.08d: %As.type.d0a = facet_value Core.IntLiteral, (%As.impl_witness.f4f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2f2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u32, %As.facet.08d) [concrete]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %As.WithSelf.Convert.type.2f2, %As.facet.08d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.82d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.0bc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.536: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.0bc, @Core.IntLiteral.as.As.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.159: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.536 [concrete]
// CHECK:STDOUT:   %int_1.b0b: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %carbon_unsigned_int.patt: %pattern_type.129 = value_binding_pattern carbon_unsigned_int [concrete]
// CHECK:STDOUT:   %f32.3bc: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.239: type = pattern_type %f32.3bc [concrete]
// CHECK:STDOUT:   %cpp_float.patt: %pattern_type.239 = value_binding_pattern cpp_float [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.f72: type = facet_type <@As, @As(%f32.3bc)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.4b6: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.0cf: %Core.FloatLiteral.as.As.impl.Convert.type.4b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ceb: <witness> = impl_witness imports.%As.impl_witness_table.591, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.40d: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.4d5: %Core.FloatLiteral.as.As.impl.Convert.type.40d = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c92: %As.type.f72 = facet_value Core.FloatLiteral, (%As.impl_witness.ceb) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ce5: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.3bc, %As.facet.c92) [concrete]
// CHECK:STDOUT:   %.6e3: type = fn_type_with_self_type %As.WithSelf.Convert.type.ce5, %As.facet.c92 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.81a: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.4d5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.4bb: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.4d5, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.88a: <bound method> = bound_method %float.6daae3.1, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.4bb [concrete]
// CHECK:STDOUT:   %float.239: %f32.3bc = float_value 1 [concrete]
// CHECK:STDOUT:   %carbon_float.patt: %pattern_type.239 = value_binding_pattern carbon_float [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %cpp_double.patt: %pattern_type.667 = value_binding_pattern cpp_double [concrete]
// CHECK:STDOUT:   %float.6daae3.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %As.type.b48: type = facet_type <@As, @As(%f64.761)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.553: <witness> = impl_witness imports.%As.impl_witness_table.591, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ecc: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.882: %Core.FloatLiteral.as.As.impl.Convert.type.ecc = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.767: %As.type.b48 = facet_value Core.FloatLiteral, (%As.impl_witness.553) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7fd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.761, %As.facet.767) [concrete]
// CHECK:STDOUT:   %.dac: type = fn_type_with_self_type %As.WithSelf.Convert.type.7fd, %As.facet.767 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound.c4d: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.882 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn.4c2: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.882, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.a68: <bound method> = bound_method %float.6daae3.2, %Core.FloatLiteral.as.As.impl.Convert.specific_fn.4c2 [concrete]
// CHECK:STDOUT:   %float.a71: %f64.761 = float_value 1 [concrete]
// CHECK:STDOUT:   %carbon_double.patt: %pattern_type.667 = value_binding_pattern carbon_double [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.753.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.753.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl.753 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d9c: @Core.IntLiteral.as.As.impl.d83.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.16e) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.d83.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.81c)]
// CHECK:STDOUT:   %As.impl_witness_table.43c = impl_witness_table (%Core.import_ref.d9c), @Core.IntLiteral.as.As.impl.d83 [concrete]
// CHECK:STDOUT:   %Core.import_ref.e66: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.4b6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.0cf)]
// CHECK:STDOUT:   %As.impl_witness_table.591 = impl_witness_table (%Core.import_ref.e66), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i8.loc8: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.706 = impl_witness_access constants.%As.impl_witness.eaa, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.af7]
// CHECK:STDOUT:   %bound_method.loc8_45.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.fa4]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.730]
// CHECK:STDOUT:   %bound_method.loc8_45.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.7a1f]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i8 = call %bound_method.loc8_45.2(%int_1.loc8) [concrete = constants.%int_1.2eb]
// CHECK:STDOUT:   %.loc8_45.1: %i8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.2eb]
// CHECK:STDOUT:   %.loc8_45.2: %i8 = converted %int_1.loc8, %.loc8_45.1 [concrete = constants.%int_1.2eb]
// CHECK:STDOUT:   %.loc8_28: type = splice_block %signed_char.ref [concrete = constants.%i8] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %signed_char.ref: type = name_ref signed_char, %i8.2 [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char: %i8 = wrapper_binding cpp_signed_char, %.loc8_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_signed_char.patt: %pattern_type.060 = value_binding_pattern cpp_signed_char [concrete = constants.%cpp_signed_char.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_signed_char.ref: %i8 = name_ref cpp_signed_char, %cpp_signed_char
// CHECK:STDOUT:   %i8.loc9: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %carbon_signed_char: %i8 = wrapper_binding carbon_signed_char, %cpp_signed_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_signed_char.patt: %pattern_type.060 = value_binding_pattern carbon_signed_char [concrete = constants.%carbon_signed_char.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16.loc11: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc11: %.f79 = impl_witness_access constants.%As.impl_witness.4de, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.262]
// CHECK:STDOUT:   %bound_method.loc11_33.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.d94]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc11_33.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.1e6]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %i16 = call %bound_method.loc11_33.2(%int_1.loc11) [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc11_33.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc11_33.2: %i16 = converted %int_1.loc11, %.loc11_33.1 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc11_22: type = splice_block %short.ref [concrete = constants.%i16] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %short.ref: type = name_ref short, %i16.2 [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short: %i16 = wrapper_binding cpp_short, %.loc11_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_short.patt: %pattern_type.9bc = value_binding_pattern cpp_short [concrete = constants.%cpp_short.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_short.ref: %i16 = name_ref cpp_short, %cpp_short
// CHECK:STDOUT:   %i16.loc12: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %carbon_short: %i16 = wrapper_binding carbon_short, %cpp_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_short.patt: %pattern_type.9bc = value_binding_pattern carbon_short [concrete = constants.%carbon_short.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc14_29.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.022]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.7fd]
// CHECK:STDOUT:   %bound_method.loc14_29.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14 [concrete = constants.%bound_method.5c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_29.2(%int_1.loc14) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc14 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc14_20: type = splice_block %int.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %int.ref: type = name_ref int, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int: %i32 = wrapper_binding cpp_int, %.loc14_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_int.patt: %pattern_type.d94 = value_binding_pattern cpp_int [concrete = constants.%cpp_int.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_int.ref: %i32 = name_ref cpp_int, %cpp_int
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_int: %i32 = wrapper_binding carbon_int, %cpp_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_int.patt: %pattern_type.d94 = value_binding_pattern carbon_int [concrete = constants.%carbon_int.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u8.loc17: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %impl.elem0.loc17: %.39f = impl_witness_access constants.%As.impl_witness.a43, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.ff3]
// CHECK:STDOUT:   %bound_method.loc17_49.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.002]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.92f]
// CHECK:STDOUT:   %bound_method.loc17_49.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.7a1a]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc17: init %u8 = call %bound_method.loc17_49.2(%int_1.loc17) [concrete = constants.%int_1.5dc]
// CHECK:STDOUT:   %.loc17_49.1: %u8 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc17 [concrete = constants.%int_1.5dc]
// CHECK:STDOUT:   %.loc17_49.2: %u8 = converted %int_1.loc17, %.loc17_49.1 [concrete = constants.%int_1.5dc]
// CHECK:STDOUT:   %.loc17_30: type = splice_block %unsigned_char.ref [concrete = constants.%u8] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_char.ref: type = name_ref unsigned_char, %u8.2 [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char: %u8 = wrapper_binding cpp_unsigned_char, %.loc17_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_char.patt: %pattern_type.e53 = value_binding_pattern cpp_unsigned_char [concrete = constants.%cpp_unsigned_char.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_char.ref: %u8 = name_ref cpp_unsigned_char, %cpp_unsigned_char
// CHECK:STDOUT:   %u8.loc18: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %carbon_unsigned_char: %u8 = wrapper_binding carbon_unsigned_char, %cpp_unsigned_char.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_char.patt: %pattern_type.e53 = value_binding_pattern carbon_unsigned_char [concrete = constants.%carbon_unsigned_char.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u16.loc20: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %impl.elem0.loc20: %.5ac = impl_witness_access constants.%As.impl_witness.8df, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.ae8]
// CHECK:STDOUT:   %bound_method.loc20_51.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.db4]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.a7d]
// CHECK:STDOUT:   %bound_method.loc20_51.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.bd4]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc20: init %u16 = call %bound_method.loc20_51.2(%int_1.loc20) [concrete = constants.%int_1.f2e]
// CHECK:STDOUT:   %.loc20_51.1: %u16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc20 [concrete = constants.%int_1.f2e]
// CHECK:STDOUT:   %.loc20_51.2: %u16 = converted %int_1.loc20, %.loc20_51.1 [concrete = constants.%int_1.f2e]
// CHECK:STDOUT:   %.loc20_31: type = splice_block %unsigned_short.ref [concrete = constants.%u16] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_short.ref: type = name_ref unsigned_short, %u16.2 [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short: %u16 = wrapper_binding cpp_unsigned_short, %.loc20_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_short.patt: %pattern_type.bc6 = value_binding_pattern cpp_unsigned_short [concrete = constants.%cpp_unsigned_short.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_short.ref: %u16 = name_ref cpp_unsigned_short, %cpp_unsigned_short
// CHECK:STDOUT:   %u16.loc21: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %carbon_unsigned_short: %u16 = wrapper_binding carbon_unsigned_short, %cpp_unsigned_short.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_short.patt: %pattern_type.bc6 = value_binding_pattern carbon_unsigned_short [concrete = constants.%carbon_unsigned_short.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32.loc23: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc23: %.a11 = impl_witness_access constants.%As.impl_witness.f4f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.0bc]
// CHECK:STDOUT:   %bound_method.loc23_47.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.82d]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.536]
// CHECK:STDOUT:   %bound_method.loc23_47.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.159]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc23: init %u32 = call %bound_method.loc23_47.2(%int_1.loc23) [concrete = constants.%int_1.b0b]
// CHECK:STDOUT:   %.loc23_47.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc23 [concrete = constants.%int_1.b0b]
// CHECK:STDOUT:   %.loc23_47.2: %u32 = converted %int_1.loc23, %.loc23_47.1 [concrete = constants.%int_1.b0b]
// CHECK:STDOUT:   %.loc23_29: type = splice_block %unsigned_int.ref [concrete = constants.%u32] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_int.ref: type = name_ref unsigned_int, %u32.2 [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int: %u32 = wrapper_binding cpp_unsigned_int, %.loc23_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_int.patt: %pattern_type.129 = value_binding_pattern cpp_unsigned_int [concrete = constants.%cpp_unsigned_int.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_int.ref: %u32 = name_ref cpp_unsigned_int, %cpp_unsigned_int
// CHECK:STDOUT:   %u32.loc24: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %carbon_unsigned_int: %u32 = wrapper_binding carbon_unsigned_int, %cpp_unsigned_int.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_int.patt: %pattern_type.129 = value_binding_pattern carbon_unsigned_int [concrete = constants.%carbon_unsigned_int.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc26: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %f32.loc26: type = type_literal constants.%f32.3bc [concrete = constants.%f32.3bc]
// CHECK:STDOUT:   %impl.elem0.loc26: %.6e3 = impl_witness_access constants.%As.impl_witness.ceb, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.4d5]
// CHECK:STDOUT:   %bound_method.loc26_35.1: <bound method> = bound_method %float.loc26, %impl.elem0.loc26 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.81a]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.4bb]
// CHECK:STDOUT:   %bound_method.loc26_35.2: <bound method> = bound_method %float.loc26, %specific_fn.loc26 [concrete = constants.%bound_method.88a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc26: init %f32.3bc = call %bound_method.loc26_35.2(%float.loc26) [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc26_35.1: %f32.3bc = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc26 [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc26_35.2: %f32.3bc = converted %float.loc26, %.loc26_35.1 [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc26_22: type = splice_block %float.ref [concrete = constants.%f32.3bc] {
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %float.ref: type = name_ref float, %f32.2 [concrete = constants.%f32.3bc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float: %f32.3bc = wrapper_binding cpp_float, %.loc26_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_float.patt: %pattern_type.239 = value_binding_pattern cpp_float [concrete = constants.%cpp_float.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_float.ref: %f32.3bc = name_ref cpp_float, %cpp_float
// CHECK:STDOUT:   %f32.loc27: type = type_literal constants.%f32.3bc [concrete = constants.%f32.3bc]
// CHECK:STDOUT:   %carbon_float: %f32.3bc = wrapper_binding carbon_float, %cpp_float.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_float.patt: %pattern_type.239 = value_binding_pattern carbon_float [concrete = constants.%carbon_float.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.loc29: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %f64.loc29: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %impl.elem0.loc29: %.dac = impl_witness_access constants.%As.impl_witness.553, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.882]
// CHECK:STDOUT:   %bound_method.loc29_37.1: <bound method> = bound_method %float.loc29, %impl.elem0.loc29 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound.c4d]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn.4c2]
// CHECK:STDOUT:   %bound_method.loc29_37.2: <bound method> = bound_method %float.loc29, %specific_fn.loc29 [concrete = constants.%bound_method.a68]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call.loc29: init %f64.761 = call %bound_method.loc29_37.2(%float.loc29) [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc29_37.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call.loc29 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc29_37.2: %f64.761 = converted %float.loc29, %.loc29_37.1 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc29_23: type = splice_block %double.ref [concrete = constants.%f64.761] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %double.ref: type = name_ref double, %f64.2 [concrete = constants.%f64.761]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double: %f64.761 = wrapper_binding cpp_double, %.loc29_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_double.patt: %pattern_type.667 = value_binding_pattern cpp_double [concrete = constants.%cpp_double.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_double.ref: %f64.761 = name_ref cpp_double, %cpp_double
// CHECK:STDOUT:   %f64.loc30: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %carbon_double: %f64.761 = wrapper_binding carbon_double, %cpp_double.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_double.patt: %pattern_type.667 = value_binding_pattern carbon_double [concrete = constants.%carbon_double.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.b48: type = facet_type <@As, @As(%f64.761)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.4b6: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.0cf: %Core.FloatLiteral.as.As.impl.Convert.type.4b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.553: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ecc: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.882: %Core.FloatLiteral.as.As.impl.Convert.type.ecc = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b48 = facet_value Core.FloatLiteral, (%As.impl_witness.553) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7fd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.761, %As.facet) [concrete]
// CHECK:STDOUT:   %.dac: type = fn_type_with_self_type %As.WithSelf.Convert.type.7fd, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.882 [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.882, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a71: %f64.761 = float_value 1 [concrete]
// CHECK:STDOUT:   %carbon_long_double.patt: %pattern_type.667 = value_binding_pattern carbon_long_double [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e66: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.4b6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.0cf)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.e66), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %f64.loc11: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %impl.elem0: %.dac = impl_witness_access constants.%As.impl_witness.553, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.882]
// CHECK:STDOUT:   %bound_method.loc11_47.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_47.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.761 = call %bound_method.loc11_47.2(%float) [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc11_47.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc11_47.2: %f64.761 = converted %float, %.loc11_47.1 [concrete = constants.%float.a71]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_long_double: <error> = wrapper_binding cpp_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_double.patt: <error> = value_binding_pattern cpp_long_double [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_double.ref: <error> = name_ref cpp_long_double, %cpp_long_double [concrete = <error>]
// CHECK:STDOUT:   %f64.loc12: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %carbon_long_double: %f64.761 = wrapper_binding carbon_long_double, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_double.patt: %pattern_type.667 = value_binding_pattern carbon_long_double [concrete = constants.%carbon_long_double.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- override_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %unsigned_int: type = class_type @unsigned_int [concrete]
// CHECK:STDOUT:   %pattern_type.990: type = pattern_type %unsigned_int [concrete]
// CHECK:STDOUT:   %unsigned_int.patt: %pattern_type.990 = ref_binding_pattern unsigned_int [concrete]
// CHECK:STDOUT:   %unsigned_int.var_patt: %pattern_type.990 = var_pattern %unsigned_int.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.4da: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.79f: %T.as_type.as.DefaultOrUnformed.impl.Op.type.4da = struct_value () [symbolic]
// CHECK:STDOUT:   %unsigned_int.Op.type: type = fn_type @unsigned_int.Op [concrete]
// CHECK:STDOUT:   %unsigned_int.Op: %unsigned_int.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%unsigned_int.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.d96: %Default.type = facet_value %unsigned_int, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.3a6: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.d96) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %unsigned_int, (%DefaultOrUnformed.impl_witness.3a6) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.129 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.type: type = cpp_overload_set_type @unsigned_int.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.type: type = fn_type @unsigned_int.foo [concrete]
// CHECK:STDOUT:   %unsigned_int.foo: %unsigned_int.foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3ea: %SubobjectDestroy.type = facet_value %unsigned_int, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b76: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3ea) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %unsigned_int, (%DestroyImpl.impl_witness.b76, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_int = %unsigned_int.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.decl: type = class_decl @unsigned_int [concrete = constants.%unsigned_int] {} {}
// CHECK:STDOUT:   %Core.import_ref.288: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.4da) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.79f)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.288), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %unsigned_int.foo.cpp_overload_set.value: %unsigned_int.foo.cpp_overload_set.type = cpp_overload_set_value @unsigned_int.foo.cpp_overload_set [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %unsigned_int.foo.decl: %unsigned_int.foo.type = fn_decl @unsigned_int.foo [concrete = constants.%unsigned_int.foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %unsigned_int.var: ref %unsigned_int = var_storage %unsigned_int.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%unsigned_int, (constants.%DefaultOrUnformed.impl_witness.3a6) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc12_38.1: %DefaultOrUnformed.type = converted constants.%unsigned_int, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_38.1 [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   %.loc12_38.2: type = converted %.loc12_38.1, %as_type [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_3.1: ref %unsigned_int = splice_block %unsigned_int.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %unsigned_int to %.loc12_3.1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %unsigned_int.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc12_25: type = splice_block %unsigned_int.ref.loc12 [concrete = constants.%unsigned_int] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_int.ref.loc12: type = name_ref unsigned_int, imports.%unsigned_int.decl [concrete = constants.%unsigned_int]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int: ref %unsigned_int = wrapper_binding unsigned_int, %unsigned_int.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unsigned_int.patt: %pattern_type.990 = ref_binding_pattern unsigned_int [concrete = constants.%unsigned_int.patt]
// CHECK:STDOUT:     %unsigned_int.var_patt: %pattern_type.990 = var_pattern %unsigned_int.patt [concrete = constants.%unsigned_int.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %unsigned_int.ref.loc13: ref %unsigned_int = name_ref unsigned_int, %unsigned_int
// CHECK:STDOUT:   %foo.ref: %unsigned_int.foo.cpp_overload_set.type = name_ref foo, imports.%unsigned_int.foo.cpp_overload_set.value [concrete = constants.%unsigned_int.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %unsigned_int.ref.loc13, %foo.ref
// CHECK:STDOUT:   %unsigned_int.foo.call: init %u32 = call imports.%unsigned_int.foo.decl(%unsigned_int.ref.loc13)
// CHECK:STDOUT:   %.loc13_40.1: %u32 = value_of_initializer %unsigned_int.foo.call
// CHECK:STDOUT:   %.loc13_40.2: %u32 = converted %unsigned_int.foo.call, %.loc13_40.1
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %x: %u32 = wrapper_binding x, %.loc13_40.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.129 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%unsigned_int, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%unsigned_int, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%unsigned_int, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.3: %Destroy.type = converted constants.%unsigned_int, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%unsigned_int.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12(%self.param: ref %unsigned_int) = "no_op";
// CHECK:STDOUT:
