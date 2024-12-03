// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_duplicate_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_duplicate_name.carbon

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:42: error: duplicated field name `abc` in struct type literal [StructNameDuplicate]
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:22: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                      ^~~
// CHECK:STDERR:
fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:19: error: duplicated field name `a` in struct type literal [StructNameDuplicate]
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:10: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
// CHECK:STDERR:
let v: {.a: i32, .a: i32} = {.a = 1};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:26: error: duplicated field name `def` in struct literal [StructNameDuplicate]
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:16: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
let w: i32 = {.def = 1, .def = 2}.def;

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:30: error: duplicated field name `a` in struct literal [StructNameDuplicate]
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                              ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:22: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                      ^
// CHECK:STDERR:
var x: {.a: i32} = {.a = 1, .a = 2};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:39: error: duplicated field name `b` in struct literal [StructNameDuplicate]
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                                       ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:31: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                               ^
var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};

// CHECK:STDOUT: --- fail_duplicate_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.a.1: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.2: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_16: init type = call constants.%Int(%int_32.loc18_16) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_16.1: type = value_of_initializer %int.make_type_signed.loc18_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_16.2: type = converted %int.make_type_signed.loc18_16, %.loc18_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_27: init type = call constants.%Int(%int_32.loc18_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_27.1: type = value_of_initializer %int.make_type_signed.loc18_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_27.2: type = converted %int.make_type_signed.loc18_27, %.loc18_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_36: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_36: init type = call constants.%Int(%int_32.loc18_36) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.1: type = value_of_initializer %int.make_type_signed.loc18_36 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.2: type = converted %int.make_type_signed.loc18_36, %.loc18_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_47: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_47: init type = call constants.%Int(%int_32.loc18_47) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_47.1: type = value_of_initializer %int.make_type_signed.loc18_47 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_47.2: type = converted %int.make_type_signed.loc18_47, %.loc18_47.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_56: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_56: init type = call constants.%Int(%int_32.loc18_56) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_56.1: type = value_of_initializer %int.make_type_signed.loc18_56 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_56.2: type = converted %int.make_type_signed.loc18_56, %.loc18_56.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc27_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc27_13: init type = call constants.%Int(%int_32.loc27_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_13.1: type = value_of_initializer %int.make_type_signed.loc27_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_13.2: type = converted %int.make_type_signed.loc27_13, %.loc27_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc27_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc27_22: init type = call constants.%Int(%int_32.loc27_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_22.1: type = value_of_initializer %int.make_type_signed.loc27_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_22.2: type = converted %int.make_type_signed.loc27_22, %.loc27_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc36: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc36: init type = call constants.%Int(%int_32.loc36) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.1: type = value_of_initializer %int.make_type_signed.loc36 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.2: type = converted %int.make_type_signed.loc36, %.loc36_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc45: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc45: init type = call constants.%Int(%int_32.loc45) [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_13.1: type = value_of_initializer %int.make_type_signed.loc45 [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_13.2: type = converted %int.make_type_signed.loc45, %.loc45_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template = constants.%struct_type.a.2]
// CHECK:STDOUT:   %x.var: ref %struct_type.a.2 = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int_32.loc53_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc53_13: init type = call constants.%Int(%int_32.loc53_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.1: type = value_of_initializer %int.make_type_signed.loc53_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.2: type = converted %int.make_type_signed.loc53_13, %.loc53_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc53_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc53_22: init type = call constants.%Int(%int_32.loc53_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_22.1: type = value_of_initializer %int.make_type_signed.loc53_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_22.2: type = converted %int.make_type_signed.loc53_22, %.loc53_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %i32} [template = constants.%struct_type.b.c]
// CHECK:STDOUT:   %y.var: ref %struct_type.b.c = var y
// CHECK:STDOUT:   %y: ref %struct_type.b.c = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc27: %struct_type.a.1 = struct_literal (%int_1.loc27)
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT:   %int_1.loc36: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc36: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %def.ref: <error> = name_ref def, <error> [template = <error>]
// CHECK:STDOUT:   %w: %i32 = bind_name w, <error>
// CHECK:STDOUT:   %int_1.loc45: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc45: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
