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
// CHECK:STDERR: fail_todo_import_api.carbon: error: semantics TODO: `Unsupported: Importing C++ functions that require thunks indirectly` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import library "api";

fn F() {
  //@dump-sem-ir-begin
  FooShort(8 as i16);
  FooInt(9);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_short.type: type = fn_type @foo_short [concrete]
// CHECK:STDOUT:   %foo_short: %foo_short.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %As.type.afd: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.8b6: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2ac: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.128: %Core.IntLiteral.as.As.impl.Convert.type.2ac = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.97f: <witness> = impl_witness imports.%As.impl_witness_table.ae5, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c2a: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.9b9: %Core.IntLiteral.as.As.impl.Convert.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.afd = facet_value Core.IntLiteral, (%As.impl_witness.97f) [concrete]
// CHECK:STDOUT:   %.ed4: type = fn_type_with_self_type %As.Convert.type.8b6, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.9b9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.9b9, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.1a0: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.892: %i16 = int_value 8 [concrete]
// CHECK:STDOUT:   %foo_int.type: type = fn_type @foo_int [concrete]
// CHECK:STDOUT:   %foo_int: %foo_int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ed2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ba4, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ed2) [concrete]
// CHECK:STDOUT:   %.2c2: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3e2: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.114: %i32 = int_value 9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FooShort: %foo_short.type = import_ref Main//api, FooShort, loaded [concrete = constants.%foo_short]
// CHECK:STDOUT:   %Main.FooInt: %foo_int.type = import_ref Main//api, FooInt, loaded [concrete = constants.%foo_int]
// CHECK:STDOUT:   %Core.import_ref.49e: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.2ac) = import_ref Core//prelude/parts/int, loc32_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.128)]
// CHECK:STDOUT:   %As.impl_witness_table.ae5 = impl_witness_table (%Core.import_ref.49e), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ba4 = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FooShort.ref: %foo_short.type = name_ref FooShort, imports.%Main.FooShort [concrete = constants.%foo_short]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc10: %.ed4 = impl_witness_access constants.%As.impl_witness.97f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.9b9]
// CHECK:STDOUT:   %bound_method.loc10_14.1: <bound method> = bound_method %int_8, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_14.2: <bound method> = bound_method %int_8, %specific_fn.loc10 [concrete = constants.%bound_method.1a0]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc10_14.2(%int_8) [concrete = constants.%int_8.892]
// CHECK:STDOUT:   %.loc10_14.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.892]
// CHECK:STDOUT:   %.loc10_14.2: %i16 = converted %int_8, %.loc10_14.1 [concrete = constants.%int_8.892]
// CHECK:STDOUT:   %foo_short.call: init %empty_tuple.type = call %FooShort.ref(%.loc10_14.2)
// CHECK:STDOUT:   %FooInt.ref: %foo_int.type = name_ref FooInt, imports.%Main.FooInt [concrete = constants.%foo_int]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %impl.elem0.loc11: %.2c2 = impl_witness_access constants.%ImplicitAs.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %int_9, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %int_9, %specific_fn.loc11 [concrete = constants.%bound_method.3e2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_10.2(%int_9) [concrete = constants.%int_9.114]
// CHECK:STDOUT:   %.loc11_10.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_9.114]
// CHECK:STDOUT:   %.loc11_10.2: %i32 = converted %int_9, %.loc11_10.1 [concrete = constants.%int_9.114]
// CHECK:STDOUT:   %foo_int.call: init %empty_tuple.type = call %FooInt.ref(%.loc11_10.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
