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

import Cpp inline "";

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

import Cpp inline "";

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
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.565: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.dc4(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.d2c: %Core.IntLiteral.as.As.impl.Convert.type.565 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.4f1: <witness> = impl_witness imports.%As.impl_witness_table.5ad, @Core.IntLiteral.as.As.impl.dc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.9ac: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.dc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.fe2: %Core.IntLiteral.as.As.impl.Convert.type.9ac = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.888: %As.type.bbb = facet_value Core.IntLiteral, (%As.impl_witness.4f1) [concrete]
// CHECK:STDOUT:   %.b12: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet.888 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.0c0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.fe2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.96b: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.fe2, @Core.IntLiteral.as.As.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.b11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.96b [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.972: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.cc4(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.276: %Core.IntLiteral.as.As.impl.Convert.type.972 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.feb: <witness> = impl_witness imports.%As.impl_witness_table.e4e, @Core.IntLiteral.as.As.impl.cc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.da0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.cc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.467: %Core.IntLiteral.as.As.impl.Convert.type.da0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.218: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.feb) [concrete]
// CHECK:STDOUT:   %.e88: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet.218 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.467 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.467, @Core.IntLiteral.as.As.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.290: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5 [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.1
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.99c: @Core.IntLiteral.as.As.impl.dc4.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.565) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.dc4.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.d2c)]
// CHECK:STDOUT:   %As.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.99c), @Core.IntLiteral.as.As.impl.dc4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.611: @Core.IntLiteral.as.As.impl.cc4.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.972) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.cc4.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.276)]
// CHECK:STDOUT:   %As.impl_witness_table.e4e = impl_witness_table (%Core.import_ref.611), @Core.IntLiteral.as.As.impl.cc4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.95b = binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc8: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc8: %.b12 = impl_witness_access constants.%As.impl_witness.4f1, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.fe2]
// CHECK:STDOUT:   %bound_method.loc8_41.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.0c0]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.96b]
// CHECK:STDOUT:   %bound_method.loc8_41.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.b11]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i64 = call %bound_method.loc8_41.2(%int_1.loc8) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_41.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_41.2: %i64 = converted %int_1.loc8, %.loc8_41.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_26: type = splice_block %long_long.ref [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = bind_name cpp_long_long, %.loc8_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc9: type = splice_block %i64.loc9 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long_long: %i64 = bind_name carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.157 = binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc11: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc11: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc11: %.e88 = impl_witness_access constants.%As.impl_witness.feb, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.467]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.fec]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.290]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc11: init %u64 = call %bound_method.loc11_59.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_59.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc11 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_59.2: %u64 = converted %int_1.loc11, %.loc11_59.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc11_35: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = bind_name cpp_unsigned_long_long, %.loc11_59.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc12: type = splice_block %u64.loc12 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc12: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc12: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = bind_name carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
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
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.565: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.dc4(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.d2c: %Core.IntLiteral.as.As.impl.Convert.type.565 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.4f1: <witness> = impl_witness imports.%As.impl_witness_table.5ad, @Core.IntLiteral.as.As.impl.dc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.9ac: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.dc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.fe2: %Core.IntLiteral.as.As.impl.Convert.type.9ac = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.888: %As.type.bbb = facet_value Core.IntLiteral, (%As.impl_witness.4f1) [concrete]
// CHECK:STDOUT:   %.b12: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet.888 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.0c0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.fe2 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.96b: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.fe2, @Core.IntLiteral.as.As.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.b11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.96b [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.972: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.cc4(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.276: %Core.IntLiteral.as.As.impl.Convert.type.972 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.feb: <witness> = impl_witness imports.%As.impl_witness_table.e4e, @Core.IntLiteral.as.As.impl.cc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.da0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.cc4(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.467: %Core.IntLiteral.as.As.impl.Convert.type.da0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.218: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.feb) [concrete]
// CHECK:STDOUT:   %.e88: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet.218 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.467 [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.467, @Core.IntLiteral.as.As.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.290: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5 [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.99c: @Core.IntLiteral.as.As.impl.dc4.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.565) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.dc4.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.d2c)]
// CHECK:STDOUT:   %As.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.99c), @Core.IntLiteral.as.As.impl.dc4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.611: @Core.IntLiteral.as.As.impl.cc4.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.972) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.cc4.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.276)]
// CHECK:STDOUT:   %As.impl_witness_table.e4e = impl_witness_table (%Core.import_ref.611), @Core.IntLiteral.as.As.impl.cc4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: <error> = binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc15: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc15: %.b12 = impl_witness_access constants.%As.impl_witness.4f1, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.fe2]
// CHECK:STDOUT:   %bound_method.loc15_31.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.0c0]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.96b]
// CHECK:STDOUT:   %bound_method.loc15_31.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.b11]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_31.2(%int_1.loc15) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_31.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc15 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_31.2: %i64 = converted %int_1.loc15, %.loc15_31.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_long: <error> = bind_name cpp_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.95b = binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: <error> = name_ref cpp_long, %cpp_long [concrete = <error>]
// CHECK:STDOUT:   %.loc16: type = splice_block %i64.loc16 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc16: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long: %i64 = bind_name carbon_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: <error> = binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc25: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc25: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc25: %.e88 = impl_witness_access constants.%As.impl_witness.feb, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.467]
// CHECK:STDOUT:   %bound_method.loc25_49.1: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.fec]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.6f5]
// CHECK:STDOUT:   %bound_method.loc25_49.2: <bound method> = bound_method %int_1.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.290]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc25: init %u64 = call %bound_method.loc25_49.2(%int_1.loc25) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc25_49.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc25 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc25_49.2: %u64 = converted %int_1.loc25, %.loc25_49.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_unsigned_long: <error> = bind_name cpp_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.157 = binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: <error> = name_ref cpp_unsigned_long, %cpp_unsigned_long [concrete = <error>]
// CHECK:STDOUT:   %.loc26: type = splice_block %u64.loc26 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc26: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc26: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = bind_name carbon_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
