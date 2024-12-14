// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/resolve_used.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/resolve_used.carbon

// --- fail_todo_call_monomorphization_error.carbon

library "[[@TEST_NAME]]";

fn ErrorIfNIsZero(N:! Core.IntLiteral(), M:! Core.IntLiteral()) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+6]]:10: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var v: Core.Int(N);
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+3]]:10: note: in `i0` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   var v: Core.Int(N);
  // CHECK:STDERR:          ^~~~~~~~~~~
  var v: Core.Int(N);
  var w: Core.Int(M);
}

fn CallNegative() {
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+3]]:3: note: in `ErrorIfNIsZero(0, 1)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   ErrorIfNIsZero(0, 1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  ErrorIfNIsZero(0, 1);
}

// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = bind_symbolic_name M, 1 [symbolic]
// CHECK:STDOUT:   %M.patt: Core.IntLiteral = symbolic_binding_pattern M, 1 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [template]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [template]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %Int.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int.1 [symbolic]
// CHECK:STDOUT:   %Int.2: type = class_type @Int, @Int(%M) [symbolic]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %Int.2 [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [template]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0, %int_1) [template]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness <error> [template]
// CHECK:STDOUT:   %i1: type = class_type @Int, @Int(%int_1) [template]
// CHECK:STDOUT:   %i1.builtin: type = int_type signed, %int_1 [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %i1.builtin [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Core//prelude/types/int, Int, loaded [template = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ErrorIfNIsZero = %ErrorIfNIsZero.decl
// CHECK:STDOUT:     .CallNegative = %CallNegative.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ErrorIfNIsZero.decl: %ErrorIfNIsZero.type = fn_decl @ErrorIfNIsZero [template = constants.%ErrorIfNIsZero] {
// CHECK:STDOUT:     %N.patt.loc6_19.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_19.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_19.1, runtime_param<invalid> [symbolic = %N.patt.loc6_19.2 (constants.%N.patt)]
// CHECK:STDOUT:     %M.patt.loc6_42.1: Core.IntLiteral = symbolic_binding_pattern M, 1 [symbolic = %M.patt.loc6_42.2 (constants.%M.patt)]
// CHECK:STDOUT:     %M.param_patt: Core.IntLiteral = value_param_pattern %M.patt.loc6_42.1, runtime_param<invalid> [symbolic = %M.patt.loc6_42.2 (constants.%M.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc6_23: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc6_27: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc6_39: init type = call %IntLiteral.ref.loc6_27() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_39.1: type = value_of_initializer %int_literal.make_type.loc6_39 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_39.2: type = converted %int_literal.make_type.loc6_39, %.loc6_39.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc6_46: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc6_50: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc6_62: init type = call %IntLiteral.ref.loc6_50() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_62.1: type = value_of_initializer %int_literal.make_type.loc6_62 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_62.2: type = converted %int_literal.make_type.loc6_62, %.loc6_62.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_19.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_19.2 (constants.%N)]
// CHECK:STDOUT:     %M.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %M.loc6_42.1: Core.IntLiteral = bind_symbolic_name M, 1, %M.param [symbolic = %M.loc6_42.2 (constants.%M)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [template = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc6_19.1: Core.IntLiteral, %M.loc6_42.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_19.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_19.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_19.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_19.2 (constants.%N.patt)]
// CHECK:STDOUT:   %M.loc6_42.2: Core.IntLiteral = bind_symbolic_name M, 1 [symbolic = %M.loc6_42.2 (constants.%M)]
// CHECK:STDOUT:   %M.patt.loc6_42.2: Core.IntLiteral = symbolic_binding_pattern M, 1 [symbolic = %M.patt.loc6_42.2 (constants.%M.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc11_20.2: type = class_type @Int, @Int(%N.loc6_19.2) [symbolic = %Int.loc11_20.2 (constants.%Int.1)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type @ErrorIfNIsZero.%Int.loc11_20.2 (%Int.1) [symbolic = %require_complete.loc11 (constants.%require_complete.2)]
// CHECK:STDOUT:   %Int.loc12_20.2: type = class_type @Int, @Int(%M.loc6_42.2) [symbolic = %Int.loc12_20.2 (constants.%Int.2)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type @ErrorIfNIsZero.%Int.loc12_20.2 (%Int.2) [symbolic = %require_complete.loc12 (constants.%require_complete.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %M.param_patt: Core.IntLiteral) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc11: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc6_19.1 [symbolic = %N.loc6_19.2 (constants.%N)]
// CHECK:STDOUT:     %Int.loc11_20.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_20.2 (constants.%Int.1)]
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc11_20.2 (%Int.1) = var v
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc11_20.2 (%Int.1) = bind_name v, %v.var
// CHECK:STDOUT:     %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc12: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int.generic]
// CHECK:STDOUT:     %M.ref: Core.IntLiteral = name_ref M, %M.loc6_42.1 [symbolic = %M.loc6_42.2 (constants.%M)]
// CHECK:STDOUT:     %Int.loc12_20.1: type = class_type @Int, @Int(constants.%M) [symbolic = %Int.loc12_20.2 (constants.%Int.2)]
// CHECK:STDOUT:     %w.var: ref @ErrorIfNIsZero.%Int.loc12_20.2 (%Int.2) = var w
// CHECK:STDOUT:     %w: ref @ErrorIfNIsZero.%Int.loc12_20.2 (%Int.2) = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallNegative() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ErrorIfNIsZero.ref: %ErrorIfNIsZero.type = name_ref ErrorIfNIsZero, file.%ErrorIfNIsZero.decl [template = constants.%ErrorIfNIsZero]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero.ref, @ErrorIfNIsZero(constants.%int_0, constants.%int_1) [template = constants.%ErrorIfNIsZero.specific_fn]
// CHECK:STDOUT:   %ErrorIfNIsZero.call: init %empty_tuple.type = call %ErrorIfNIsZero.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N, constants.%M) {
// CHECK:STDOUT:   %N.loc6_19.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_19.2 => constants.%N
// CHECK:STDOUT:   %M.loc6_42.2 => constants.%M
// CHECK:STDOUT:   %M.patt.loc6_42.2 => constants.%M
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0, constants.%int_1) {
// CHECK:STDOUT:   %N.loc6_19.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc6_19.2 => constants.%int_0
// CHECK:STDOUT:   %M.loc6_42.2 => constants.%int_1
// CHECK:STDOUT:   %M.patt.loc6_42.2 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc11_20.2 => constants.%i0
// CHECK:STDOUT:   %require_complete.loc11 => constants.%complete_type.3
// CHECK:STDOUT:   %Int.loc12_20.2 => constants.%i1
// CHECK:STDOUT:   %require_complete.loc12 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
