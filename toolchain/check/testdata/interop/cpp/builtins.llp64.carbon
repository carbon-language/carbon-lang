// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// EXTRA-ARGS: --target=x86_64-pc-windows-msvc --clang-arg=-fno-PIE
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon

// --- supported_types.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_long_long : Cpp.long_long = 1 as i64;
  let carbon_long_long: i64 = cpp_long_long;

  let cpp_unsigned_long_long : Cpp.unsigned_long_long = 1 as u64;
  let carbon_unsigned_long_long: u64 = cpp_unsigned_long_long;
  //@dump-sem-ir-end
}

// --- fail_todo_unsupported_types.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+7]]:18: error: semantics TODO: `Unsupported: builtin type: long` [SemanticsTodo]
  // CHECK:STDERR:   let cpp_long : Cpp.long = 1 as i64;
  // CHECK:STDERR:                  ^~~~~~~~
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+4]]:18: note: in `Cpp` name lookup for `long` [InCppNameLookup]
  // CHECK:STDERR:   let cpp_long : Cpp.long = 1 as i64;
  // CHECK:STDERR:                  ^~~~~~~~
  // CHECK:STDERR:
  let cpp_long : Cpp.long = 1 as i64;
  let carbon_long: i64 = cpp_long;

  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+7]]:27: error: semantics TODO: `Unsupported: builtin type: unsigned long` [SemanticsTodo]
  // CHECK:STDERR:   let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  // CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+4]]:27: note: in `Cpp` name lookup for `unsigned_long` [InCppNameLookup]
  // CHECK:STDERR:   let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  // CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  let carbon_unsigned_long: u64 = cpp_unsigned_long;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- supported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.bbb: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c40: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4f0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.108: %Core.IntLiteral.as.As.impl.Convert.type.4f0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.2cd: %As.type.bbb = facet_value Core.IntLiteral, (%As.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %.277: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet.2cd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.373: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.108 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.108, @Core.IntLiteral.as.As.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c84: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.56b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.02d: %Core.IntLiteral.as.As.impl.Convert.type.56b = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ed2: <witness> = impl_witness imports.%As.impl_witness_table.a7d, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.422: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a09: %Core.IntLiteral.as.As.impl.Convert.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.bf5: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.ed2) [concrete]
// CHECK:STDOUT:   %.5b8: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet.bf5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.eb8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.009: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a09, @Core.IntLiteral.as.As.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.bb9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.009 [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.1
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl.863 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.56b) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.02d)]
// CHECK:STDOUT:   %As.impl_witness_table.a7d = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.As.impl.38a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.95b = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc8: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc8: %.277 = impl_witness_access constants.%As.impl_witness.c40, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.108]
// CHECK:STDOUT:   %bound_method.loc8_41.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.373]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.83e]
// CHECK:STDOUT:   %bound_method.loc8_41.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.c84]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i64 = call %bound_method.loc8_41.2(%int_1.loc8) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_41.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_41.2: %i64 = converted %int_1.loc8, %.loc8_41.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_26: type = splice_block %long_long.ref [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc8_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc9: type = splice_block %i64.loc9 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc11: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc11: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc11: %.5b8 = impl_witness_access constants.%As.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a09]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.eb8]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.009]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.bb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %u64 = call %bound_method.loc11_59.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_59.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_59.2: %u64 = converted %int_1.loc11, %.loc11_59.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_35: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc11_59.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc12: type = splice_block %u64.loc12 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc12: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc12: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.bbb: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c40: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4f0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.108: %Core.IntLiteral.as.As.impl.Convert.type.4f0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.2cd: %As.type.bbb = facet_value Core.IntLiteral, (%As.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %.277: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet.2cd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.373: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.108 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.108, @Core.IntLiteral.as.As.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c84: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.56b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.02d: %Core.IntLiteral.as.As.impl.Convert.type.56b = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ed2: <witness> = impl_witness imports.%As.impl_witness_table.a7d, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.422: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a09: %Core.IntLiteral.as.As.impl.Convert.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.bf5: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.ed2) [concrete]
// CHECK:STDOUT:   %.5b8: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet.bf5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.eb8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a09 [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.009: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a09, @Core.IntLiteral.as.As.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.bb9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.009 [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl.863 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.56b) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.02d)]
// CHECK:STDOUT:   %As.impl_witness_table.a7d = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.As.impl.38a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: <error> = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc15: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc15: %.277 = impl_witness_access constants.%As.impl_witness.c40, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.108]
// CHECK:STDOUT:   %bound_method.loc15_31.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.373]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.83e]
// CHECK:STDOUT:   %bound_method.loc15_31.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.c84]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_31.2(%int_1.loc15) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_31.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc15 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_31.2: %i64 = converted %int_1.loc15, %.loc15_31.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_long: <error> = value_binding cpp_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.95b = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: <error> = name_ref cpp_long, %cpp_long [concrete = <error>]
// CHECK:STDOUT:   %.loc16: type = splice_block %i64.loc16 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc16: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: <error> = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc25: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc25: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc25: %.5b8 = impl_witness_access constants.%As.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a09]
// CHECK:STDOUT:   %bound_method.loc25_49.1: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.eb8]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.009]
// CHECK:STDOUT:   %bound_method.loc25_49.2: <bound method> = bound_method %int_1.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.bb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc25: init %u64 = call %bound_method.loc25_49.2(%int_1.loc25) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc25_49.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc25 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc25_49.2: %u64 = converted %int_1.loc25, %.loc25_49.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_unsigned_long: <error> = value_binding cpp_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: <error> = name_ref cpp_unsigned_long, %cpp_unsigned_long [concrete = <error>]
// CHECK:STDOUT:   %.loc26: type = splice_block %u64.loc26 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc26: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc26: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
