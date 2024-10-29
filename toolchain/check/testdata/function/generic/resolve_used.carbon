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

// TODO: This declaration should be accepted, and the `var v` below should be
// rejected during monomorphization, once integer literals have the type
// `Core.IntLiteral(N)` as described in the design.
// CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+3]]:41: error: name `IntLiteral` not found [NameNotFound]
// CHECK:STDERR: fn ErrorIfNIsZero[N:! Core.BigInt()](n: Core.IntLiteral(N)) {
// CHECK:STDERR:                                         ^~~~~~~~~~~~~~~
fn ErrorIfNIsZero[N:! Core.BigInt()](n: Core.IntLiteral(N)) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  var v: Core.Int(N);
}

fn CallNegative() {
  ErrorIfNIsZero(0);
}

// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BigInt.type: type = fn_type @BigInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BigInt: %BigInt.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.BigInt = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [template]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [template]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .BigInt = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %BigInt.type = import_ref Core//prelude/types, inst+7, loaded [template = constants.%BigInt]
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Core//prelude/types, inst+30, loaded [template = constants.%Int]
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
// CHECK:STDOUT:     %N.patt.loc10_19.1: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_19.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.BigInt = value_param_pattern %N.patt.loc10_19.1, runtime_param<invalid> [symbolic = %N.patt.loc10_19.2 (constants.%N.patt)]
// CHECK:STDOUT:     %n.patt: <error> = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: <error> = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc10_23: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type: init type = call %BigInt.ref() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc10_35.1: type = value_of_initializer %big_int.make_type [template = Core.BigInt]
// CHECK:STDOUT:     %.loc10_35.2: type = converted %big_int.make_type, %.loc10_35.1 [template = Core.BigInt]
// CHECK:STDOUT:     %Core.ref.loc10_41: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: <error> = name_ref IntLiteral, <error> [template = <error>]
// CHECK:STDOUT:     %N.ref.loc10: Core.BigInt = name_ref N, %N.loc10_19.1 [symbolic = %N.loc10_19.2 (constants.%N)]
// CHECK:STDOUT:     %N.param: Core.BigInt = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_19.1: Core.BigInt = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc10_19.2 (constants.%N)]
// CHECK:STDOUT:     %n.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %n: <error> = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [template = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BigInt() -> type = "big_int.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc10_19.1: Core.BigInt) {
// CHECK:STDOUT:   %N.loc10_19.2: Core.BigInt = bind_symbolic_name N, 0 [symbolic = %N.loc10_19.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc10_19.2: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_19.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc15_18: type = int_type signed, %N.loc10_19.2 [symbolic = %.loc15_18 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: Core.BigInt](%n.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc15: Core.BigInt = name_ref N, %N.loc10_19.1 [symbolic = %N.loc10_19.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%N.ref.loc15) [symbolic = %.loc15_18 (constants.%.2)]
// CHECK:STDOUT:     %.loc15_20.1: type = value_of_initializer %int.make_type_signed [symbolic = %.loc15_18 (constants.%.2)]
// CHECK:STDOUT:     %.loc15_20.2: type = converted %int.make_type_signed, %.loc15_20.1 [symbolic = %.loc15_18 (constants.%.2)]
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%.loc15_18 (%.2) = var v
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%.loc15_18 (%.2) = bind_name v, %v.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%size.param_patt: Core.BigInt) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallNegative() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ErrorIfNIsZero.ref: %ErrorIfNIsZero.type = name_ref ErrorIfNIsZero, file.%ErrorIfNIsZero.decl [template = constants.%ErrorIfNIsZero]
// CHECK:STDOUT:   %.loc19: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.loc10_19.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc10_19.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
