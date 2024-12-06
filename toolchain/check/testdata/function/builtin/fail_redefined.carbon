// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon

fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn A` [RedeclRedef]
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, m: i32) -> i32 { return n; }

fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn B` [RedeclRedef]
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+6]]:1: error: redefinition of `fn C` [RedeclRedef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .B = %B.decl.loc21
// CHECK:STDOUT:     .C = %C.decl.loc31
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_9: init type = call constants.%Int(%int_32.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_signed.loc11_9, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_17: init type = call constants.%Int(%int_32.loc11_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.1: type = value_of_initializer %int.make_type_signed.loc11_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.2: type = converted %int.make_type_signed.loc11_17, %.loc11_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_25: init type = call constants.%Int(%int_32.loc11_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_signed.loc11_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_signed.loc11_25, %.loc11_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc11: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc11: %i32 = bind_name n, %n.param.loc11
// CHECK:STDOUT:     %m.param.loc11: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc11: %i32 = bind_name m, %m.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc11: ref %i32 = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc19: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc19_9: init type = call constants.%Int(%int_32.loc19_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_9.1: type = value_of_initializer %int.make_type_signed.loc19_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_9.2: type = converted %int.make_type_signed.loc19_9, %.loc19_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc19_17: init type = call constants.%Int(%int_32.loc19_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.1: type = value_of_initializer %int.make_type_signed.loc19_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.2: type = converted %int.make_type_signed.loc19_17, %.loc19_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc19_25: init type = call constants.%Int(%int_32.loc19_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.1: type = value_of_initializer %int.make_type_signed.loc19_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.2: type = converted %int.make_type_signed.loc19_25, %.loc19_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc19: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc19: %i32 = bind_name n, %n.param.loc19
// CHECK:STDOUT:     %m.param.loc19: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc19: %i32 = bind_name m, %m.param.loc19
// CHECK:STDOUT:     %return.param.loc19: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc19: ref %i32 = return_slot %return.param.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc21: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc21_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_9: init type = call constants.%Int(%int_32.loc21_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_9.1: type = value_of_initializer %int.make_type_signed.loc21_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_9.2: type = converted %int.make_type_signed.loc21_9, %.loc21_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc21_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_17: init type = call constants.%Int(%int_32.loc21_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_17.1: type = value_of_initializer %int.make_type_signed.loc21_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_17.2: type = converted %int.make_type_signed.loc21_17, %.loc21_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc21_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_25: init type = call constants.%Int(%int_32.loc21_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_25.1: type = value_of_initializer %int.make_type_signed.loc21_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_25.2: type = converted %int.make_type_signed.loc21_25, %.loc21_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc21: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc21: %i32 = bind_name n, %n.param.loc21
// CHECK:STDOUT:     %m.param.loc21: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc21: %i32 = bind_name m, %m.param.loc21
// CHECK:STDOUT:     %return.param.loc21: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc21: ref %i32 = return_slot %return.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc29_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc29_9: init type = call constants.%Int(%int_32.loc29_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_9.1: type = value_of_initializer %int.make_type_signed.loc29_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_9.2: type = converted %int.make_type_signed.loc29_9, %.loc29_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc29_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc29_17: init type = call constants.%Int(%int_32.loc29_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_17.1: type = value_of_initializer %int.make_type_signed.loc29_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_17.2: type = converted %int.make_type_signed.loc29_17, %.loc29_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc29_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc29_25: init type = call constants.%Int(%int_32.loc29_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_25.1: type = value_of_initializer %int.make_type_signed.loc29_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_25.2: type = converted %int.make_type_signed.loc29_25, %.loc29_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc29: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc29: %i32 = bind_name n, %n.param.loc29
// CHECK:STDOUT:     %m.param.loc29: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc29: %i32 = bind_name m, %m.param.loc29
// CHECK:STDOUT:     %return.param.loc29: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc29: ref %i32 = return_slot %return.param.loc29
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc31: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc31_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_9: init type = call constants.%Int(%int_32.loc31_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_9.1: type = value_of_initializer %int.make_type_signed.loc31_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_9.2: type = converted %int.make_type_signed.loc31_9, %.loc31_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc31_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_17: init type = call constants.%Int(%int_32.loc31_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_17.1: type = value_of_initializer %int.make_type_signed.loc31_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_17.2: type = converted %int.make_type_signed.loc31_17, %.loc31_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc31_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_25: init type = call constants.%Int(%int_32.loc31_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_25.1: type = value_of_initializer %int.make_type_signed.loc31_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_25.2: type = converted %int.make_type_signed.loc31_25, %.loc31_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc31: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc31: %i32 = bind_name n, %n.param.loc31
// CHECK:STDOUT:     %m.param.loc31: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc31: %i32 = bind_name m, %m.param.loc31
// CHECK:STDOUT:     %return.param.loc31: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc31: ref %i32 = return_slot %return.param.loc31
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc38: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc38_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc38_9: init type = call constants.%Int(%int_32.loc38_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_9.1: type = value_of_initializer %int.make_type_signed.loc38_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_9.2: type = converted %int.make_type_signed.loc38_9, %.loc38_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc38_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc38_17: init type = call constants.%Int(%int_32.loc38_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_17.1: type = value_of_initializer %int.make_type_signed.loc38_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_17.2: type = converted %int.make_type_signed.loc38_17, %.loc38_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc38_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc38_25: init type = call constants.%Int(%int_32.loc38_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_25.1: type = value_of_initializer %int.make_type_signed.loc38_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_25.2: type = converted %int.make_type_signed.loc38_25, %.loc38_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc38: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc38: %i32 = bind_name n, %n.param.loc38
// CHECK:STDOUT:     %m.param.loc38: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc38: %i32 = bind_name m, %m.param.loc38
// CHECK:STDOUT:     %return.param.loc38: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return.loc38: ref %i32 = return_slot %return.param.loc38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n.param_patt: %i32, %m.param_patt: %i32) -> %i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n.loc19
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n.param_patt: %i32, %m.param_patt: %i32) -> %i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n.loc21
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%n.param_patt: %i32, %m.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
