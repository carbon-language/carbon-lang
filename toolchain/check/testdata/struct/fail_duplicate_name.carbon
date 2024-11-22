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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.b: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 4 [template]
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
// CHECK:STDOUT:     %.loc18_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18_16: init type = call constants.%Int(%.loc18_16.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_16.2: type = value_of_initializer %int.make_type_signed.loc18_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_16.3: type = converted %int.make_type_signed.loc18_16, %.loc18_16.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18_27: init type = call constants.%Int(%.loc18_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_27.2: type = value_of_initializer %int.make_type_signed.loc18_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_27.3: type = converted %int.make_type_signed.loc18_27, %.loc18_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18_36: init type = call constants.%Int(%.loc18_36.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.2: type = value_of_initializer %int.make_type_signed.loc18_36 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.3: type = converted %int.make_type_signed.loc18_36, %.loc18_36.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_47.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18_47: init type = call constants.%Int(%.loc18_47.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_47.2: type = value_of_initializer %int.make_type_signed.loc18_47 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_47.3: type = converted %int.make_type_signed.loc18_47, %.loc18_47.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_56.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18_56: init type = call constants.%Int(%.loc18_56.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_56.2: type = value_of_initializer %int.make_type_signed.loc18_56 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_56.3: type = converted %int.make_type_signed.loc18_56, %.loc18_56.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc27_13: init type = call constants.%Int(%.loc27_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_13.2: type = value_of_initializer %int.make_type_signed.loc27_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_13.3: type = converted %int.make_type_signed.loc27_13, %.loc27_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc27_22: init type = call constants.%Int(%.loc27_22.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_22.2: type = value_of_initializer %int.make_type_signed.loc27_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_22.3: type = converted %int.make_type_signed.loc27_22, %.loc27_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc36: init type = call constants.%Int(%.loc36_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.2: type = value_of_initializer %int.make_type_signed.loc36 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.3: type = converted %int.make_type_signed.loc36, %.loc36_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc45: init type = call constants.%Int(%.loc45_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_13.2: type = value_of_initializer %int.make_type_signed.loc45 [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_13.3: type = converted %int.make_type_signed.loc45, %.loc45_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc45_16: type = struct_type {.a: %i32} [template = constants.%.5]
// CHECK:STDOUT:   %x.var: ref %.5 = var x
// CHECK:STDOUT:   %x: ref %.5 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc53_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc53_13: init type = call constants.%Int(%.loc53_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.2: type = value_of_initializer %int.make_type_signed.loc53_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.3: type = converted %int.make_type_signed.loc53_13, %.loc53_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc53_22: init type = call constants.%Int(%.loc53_22.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_22.2: type = value_of_initializer %int.make_type_signed.loc53_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_22.3: type = converted %int.make_type_signed.loc53_22, %.loc53_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_25: type = struct_type {.b: %i32, .c: %i32} [template = constants.%.6]
// CHECK:STDOUT:   %y.var: ref %.6 = var y
// CHECK:STDOUT:   %y: ref %.6 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_35: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc27_36: %.3 = struct_literal (%.loc27_35)
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT:   %.loc36_22: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc36_32: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %def.ref: <error> = name_ref def, <error> [template = <error>]
// CHECK:STDOUT:   %w: %i32 = bind_name w, <error>
// CHECK:STDOUT:   %.loc45_26: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc45_34: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %.loc53_35: Core.IntLiteral = int_value 3 [template = constants.%.8]
// CHECK:STDOUT:   %.loc53_43: Core.IntLiteral = int_value 4 [template = constants.%.9]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
