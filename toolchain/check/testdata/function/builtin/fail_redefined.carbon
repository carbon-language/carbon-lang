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
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `fn A`.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, m: i32) -> i32 { return n; }

fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `fn B`.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn C`.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_32.loc11_9, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc11_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n.loc11_6.2: i32 = bind_name n, %n.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17.1: type = value_of_initializer %int.make_type_32.loc11_17 [template = i32]
// CHECK:STDOUT:     %.loc11_17.2: type = converted %int.make_type_32.loc11_17, %.loc11_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc11_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     %m.loc11_14.2: i32 = bind_name m, %m.loc11_14.1
// CHECK:STDOUT:     %int.make_type_32.loc11_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_32.loc11_25 [template = i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_32.loc11_25, %.loc11_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc11: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc19: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %int.make_type_32.loc19_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_9.1: type = value_of_initializer %int.make_type_32.loc19_9 [template = i32]
// CHECK:STDOUT:     %.loc19_9.2: type = converted %int.make_type_32.loc19_9, %.loc19_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc19_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @A.%n: i32 = bind_name n, %n.loc19_6.1
// CHECK:STDOUT:     %int.make_type_32.loc19_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_17.1: type = value_of_initializer %int.make_type_32.loc19_17 [template = i32]
// CHECK:STDOUT:     %.loc19_17.2: type = converted %int.make_type_32.loc19_17, %.loc19_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc19_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     @A.%m: i32 = bind_name m, %m.loc19_14.1
// CHECK:STDOUT:     %int.make_type_32.loc19_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_25.1: type = value_of_initializer %int.make_type_32.loc19_25 [template = i32]
// CHECK:STDOUT:     %.loc19_25.2: type = converted %int.make_type_32.loc19_25, %.loc19_25.1 [template = i32]
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc21: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc21_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_9.1: type = value_of_initializer %int.make_type_32.loc21_9 [template = i32]
// CHECK:STDOUT:     %.loc21_9.2: type = converted %int.make_type_32.loc21_9, %.loc21_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc21_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n.loc21_6.2: i32 = bind_name n, %n.loc21_6.1
// CHECK:STDOUT:     %int.make_type_32.loc21_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_17.1: type = value_of_initializer %int.make_type_32.loc21_17 [template = i32]
// CHECK:STDOUT:     %.loc21_17.2: type = converted %int.make_type_32.loc21_17, %.loc21_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc21_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     %m.loc21_14.2: i32 = bind_name m, %m.loc21_14.1
// CHECK:STDOUT:     %int.make_type_32.loc21_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_25.1: type = value_of_initializer %int.make_type_32.loc21_25 [template = i32]
// CHECK:STDOUT:     %.loc21_25.2: type = converted %int.make_type_32.loc21_25, %.loc21_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc21: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc29_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_9.1: type = value_of_initializer %int.make_type_32.loc29_9 [template = i32]
// CHECK:STDOUT:     %.loc29_9.2: type = converted %int.make_type_32.loc29_9, %.loc29_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc29_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @B.%n: i32 = bind_name n, %n.loc29_6.1
// CHECK:STDOUT:     %int.make_type_32.loc29_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_17.1: type = value_of_initializer %int.make_type_32.loc29_17 [template = i32]
// CHECK:STDOUT:     %.loc29_17.2: type = converted %int.make_type_32.loc29_17, %.loc29_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc29_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     @B.%m: i32 = bind_name m, %m.loc29_14.1
// CHECK:STDOUT:     %int.make_type_32.loc29_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_25.1: type = value_of_initializer %int.make_type_32.loc29_25 [template = i32]
// CHECK:STDOUT:     %.loc29_25.2: type = converted %int.make_type_32.loc29_25, %.loc29_25.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc31: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc31_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_9.1: type = value_of_initializer %int.make_type_32.loc31_9 [template = i32]
// CHECK:STDOUT:     %.loc31_9.2: type = converted %int.make_type_32.loc31_9, %.loc31_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc31_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n.loc31_6.2: i32 = bind_name n, %n.loc31_6.1
// CHECK:STDOUT:     %int.make_type_32.loc31_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_17.1: type = value_of_initializer %int.make_type_32.loc31_17 [template = i32]
// CHECK:STDOUT:     %.loc31_17.2: type = converted %int.make_type_32.loc31_17, %.loc31_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc31_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     %m.loc31_14.2: i32 = bind_name m, %m.loc31_14.1
// CHECK:STDOUT:     %int.make_type_32.loc31_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_25.1: type = value_of_initializer %int.make_type_32.loc31_25 [template = i32]
// CHECK:STDOUT:     %.loc31_25.2: type = converted %int.make_type_32.loc31_25, %.loc31_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc31: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc38: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc38_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_9.1: type = value_of_initializer %int.make_type_32.loc38_9 [template = i32]
// CHECK:STDOUT:     %.loc38_9.2: type = converted %int.make_type_32.loc38_9, %.loc38_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc38_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @C.%n: i32 = bind_name n, %n.loc38_6.1
// CHECK:STDOUT:     %int.make_type_32.loc38_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_17.1: type = value_of_initializer %int.make_type_32.loc38_17 [template = i32]
// CHECK:STDOUT:     %.loc38_17.2: type = converted %int.make_type_32.loc38_17, %.loc38_17.1 [template = i32]
// CHECK:STDOUT:     %m.loc38_14.1: i32 = param m, runtime_param1
// CHECK:STDOUT:     @C.%m: i32 = bind_name m, %m.loc38_14.1
// CHECK:STDOUT:     %int.make_type_32.loc38_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_25.1: type = value_of_initializer %int.make_type_32.loc38_25 [template = i32]
// CHECK:STDOUT:     %.loc38_25.2: type = converted %int.make_type_32.loc38_25, %.loc38_25.1 [template = i32]
// CHECK:STDOUT:     @C.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: i32, %m: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: i32, %m: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, file.%n.loc21_6.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%n: i32, %m: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
