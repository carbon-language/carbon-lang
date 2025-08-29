// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/import.carbon

// --- header.h

auto foo_short(short x) -> void;
auto foo_int(int x) -> void;

// --- api.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

alias FooShort = Cpp.foo_short;
alias FooInt = Cpp.foo_int;

// --- fail_todo_import_api.carbon
// CHECK:STDERR: fail_todo_import_api.carbon: error: semantics TODO: `Unsupported: Importing C++ functions that require thunks indirectly called here` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import library "api";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_api.carbon:[[@LINE+7]]:3: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   FooShort(8 as i16);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_api.carbon:[[@LINE-5]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F() {
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR:
  FooShort(8 as i16);
  // FooInt(9);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.062: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.527: %Core.IntLiteral.as.As.impl.Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.172: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.489: %Core.IntLiteral.as.As.impl.Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.489 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.489, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.823: %i16 = int_value 8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FooShort: %F.type = import_ref Main//api, FooShort, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.import_ref.78a: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FooShort.ref: %F.type = name_ref FooShort, imports.%Main.FooShort [concrete = constants.%F]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.489]
// CHECK:STDOUT:   %bound_method.loc17_14.1: <bound method> = bound_method %int_8, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.2: <bound method> = bound_method %int_8, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc17_14.2(%int_8) [concrete = constants.%int_8.823]
// CHECK:STDOUT:   %.loc17_14.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.823]
// CHECK:STDOUT:   %.loc17_14.2: %i16 = converted %int_8, %.loc17_14.1 [concrete = constants.%int_8.823]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
