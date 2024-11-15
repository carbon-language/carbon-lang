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

fn ErrorIfNIsZero(N:! Core.IntLiteral()) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  var v: Core.Int(N);
}

fn CallNegative() {
  // TODO: This should fail during monomorphization once integer literals have
  // the type `Core.IntLiteral`.
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+9]]:3: error: cannot implicitly convert from `i32` to `Core.IntLiteral` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   ErrorIfNIsZero(0);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+6]]:3: note: type `i32` does not implement interface `ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ErrorIfNIsZero(0);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE-17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn ErrorIfNIsZero(N:! Core.IntLiteral()) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ErrorIfNIsZero(0);
}

// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [template]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [template]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types, inst+7, loaded [template = constants.%IntLiteral]
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
// CHECK:STDOUT:     %N.patt.loc4_19.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_19.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc4_19.1, runtime_param<invalid> [symbolic = %N.patt.loc4_19.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_39.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_39.2: type = converted %int_literal.make_type, %.loc4_39.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_19.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [template = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_19.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_19.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_19.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_19.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc9_18: type = int_type signed, %N.loc4_19.2 [symbolic = %.loc9_18 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc4_19.1 [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%N.ref) [symbolic = %.loc9_18 (constants.%.1)]
// CHECK:STDOUT:     %.loc9_20.1: type = value_of_initializer %int.make_type_signed [symbolic = %.loc9_18 (constants.%.1)]
// CHECK:STDOUT:     %.loc9_20.2: type = converted %int.make_type_signed, %.loc9_20.1 [symbolic = %.loc9_18 (constants.%.1)]
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%.loc9_18 (%.1) = var v
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%.loc9_18 (%.1) = bind_name v, %v.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallNegative() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ErrorIfNIsZero.ref: %ErrorIfNIsZero.type = name_ref ErrorIfNIsZero, file.%ErrorIfNIsZero.decl [template = constants.%ErrorIfNIsZero]
// CHECK:STDOUT:   %.loc24_18: i32 = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc24_17: Core.IntLiteral = converted %.loc24_18, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_19.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
