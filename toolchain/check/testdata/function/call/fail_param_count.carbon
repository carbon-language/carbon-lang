// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/fail_param_count.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/fail_param_count.carbon

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-8]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  // CHECK:STDERR:
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 2 arguments passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-16]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  // CHECK:STDERR:
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-24]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-32]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 0 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-40]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run2(0);
}

// CHECK:STDOUT: --- fail_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run0.type: type = fn_type @Run0 [concrete]
// CHECK:STDOUT:   %Run0: %Run0.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Run1.type: type = fn_type @Run1 [concrete]
// CHECK:STDOUT:   %Run1: %Run1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Run2.type: type = fn_type @Run2 [concrete]
// CHECK:STDOUT:   %Run2: %Run2.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run0 = %Run0.decl
// CHECK:STDOUT:     .Run1 = %Run1.decl
// CHECK:STDOUT:     .Run2 = %Run2.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Run0.decl: %Run0.type = fn_decl @Run0 [concrete = constants.%Run0] {} {}
// CHECK:STDOUT:   %Run1.decl: %Run1.type = fn_decl @Run1 [concrete = constants.%Run1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run2.decl: %Run2.type = fn_decl @Run2 [concrete = constants.%Run2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc16_12: type = splice_block %i32.loc16_12 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc16_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc16_20: type = splice_block %i32.loc16_20 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc16_20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run0() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run1(%a.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run2(%a.param: %i32, %b.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Run0.ref.loc26: %Run0.type = name_ref Run0, file.%Run0.decl [concrete = constants.%Run0]
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Run0.ref.loc34: %Run0.type = name_ref Run0, file.%Run0.decl [concrete = constants.%Run0]
// CHECK:STDOUT:   %int_0.loc34: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Run1.ref.loc43: %Run1.type = name_ref Run1, file.%Run1.decl [concrete = constants.%Run1]
// CHECK:STDOUT:   %Run1.ref.loc51: %Run1.type = name_ref Run1, file.%Run1.decl [concrete = constants.%Run1]
// CHECK:STDOUT:   %int_0.loc51: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc51: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Run2.ref.loc60: %Run2.type = name_ref Run2, file.%Run2.decl [concrete = constants.%Run2]
// CHECK:STDOUT:   %Run2.ref.loc68: %Run2.type = name_ref Run2, file.%Run2.decl [concrete = constants.%Run2]
// CHECK:STDOUT:   %int_0.loc68: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
