// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_error_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_error_recovery.carbon

// Each of these should only produce one error, and shouldn't complain that
// `<error>` doesn't implement an operator.

fn F() {
  // CHECK:STDERR: fail_error_recovery.carbon:[[@LINE+4]]:3: error: name `undeclared` not found [NameNotFound]
  // CHECK:STDERR:   undeclared + 1;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  undeclared + 1;
}

fn G(n: i32) {
  // CHECK:STDERR: fail_error_recovery.carbon:[[@LINE+4]]:7: error: name `undeclared` not found [NameNotFound]
  // CHECK:STDERR:   n + undeclared;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  n + undeclared;
}

// CHECK:STDOUT: --- fail_error_recovery.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.52d: type = fn_type @Op.2, @impl.2c2(%N) [symbolic]
// CHECK:STDOUT:   %Op.12d: %Op.type.52d = struct_value () [symbolic]
// CHECK:STDOUT:   %Add.impl_witness.dbc: <witness> = impl_witness imports.%Add.impl_witness_table, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.210: type = fn_type @Op.2, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.c82: %Op.type.210 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32, (%Add.impl_witness.dbc) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Add: type = import_ref Core//prelude/operators/arithmetic, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.344: @impl.2c2.%Op.type (%Op.type.52d) = import_ref Core//prelude/types/int, loc58_42, loaded [symbolic = @impl.2c2.%Op (constants.%Op.12d)]
// CHECK:STDOUT:   %Add.impl_witness_table = impl_witness_table (%Core.import_ref.344), @impl.2c2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .undeclared = <poisoned>
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc22: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref: <error> = name_ref undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %undeclared.ref: <error> = name_ref undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0: %.ac9 = impl_witness_access constants.%Add.impl_witness.dbc, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
