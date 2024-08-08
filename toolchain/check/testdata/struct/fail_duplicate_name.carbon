// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_duplicate_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_duplicate_name.carbon

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:42: ERROR: Duplicated field name `abc` in struct type literal.
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:22: Field with the same name here.
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                      ^~~
// CHECK:STDERR:
fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:19: ERROR: Duplicated field name `a` in struct type literal.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:10: Field with the same name here.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
// CHECK:STDERR:
let v: {.a: i32, .a: i32} = {.a = 1};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:26: ERROR: Duplicated field name `def` in struct literal.
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:16: Field with the same name here.
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
let w: i32 = {.def = 1, .def = 2}.def;

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:30: ERROR: Duplicated field name `a` in struct literal.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                              ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:22: Field with the same name here.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                      ^
// CHECK:STDERR:
var x: {.a: i32} = {.a = 1, .a = 2};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:39: ERROR: Duplicated field name `b` in struct literal.
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                                       ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:31: Field with the same name here.
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                               ^
var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};

// CHECK:STDOUT: --- fail_duplicate_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 4 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc18_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_16.1: type = value_of_initializer %int.make_type_32.loc18_16 [template = i32]
// CHECK:STDOUT:     %.loc18_16.2: type = converted %int.make_type_32.loc18_16, %.loc18_16.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc18_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_27.1: type = value_of_initializer %int.make_type_32.loc18_27 [template = i32]
// CHECK:STDOUT:     %.loc18_27.2: type = converted %int.make_type_32.loc18_27, %.loc18_27.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc18_36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_36.1: type = value_of_initializer %int.make_type_32.loc18_36 [template = i32]
// CHECK:STDOUT:     %.loc18_36.2: type = converted %int.make_type_32.loc18_36, %.loc18_36.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc18_47: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_47.1: type = value_of_initializer %int.make_type_32.loc18_47 [template = i32]
// CHECK:STDOUT:     %.loc18_47.2: type = converted %int.make_type_32.loc18_47, %.loc18_47.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc18_56: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_56.1: type = value_of_initializer %int.make_type_32.loc18_56 [template = i32]
// CHECK:STDOUT:     %.loc18_56.2: type = converted %int.make_type_32.loc18_56, %.loc18_56.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc27_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_13.1: type = value_of_initializer %int.make_type_32.loc27_13 [template = i32]
// CHECK:STDOUT:   %.loc27_13.2: type = converted %int.make_type_32.loc27_13, %.loc27_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc27_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_22.1: type = value_of_initializer %int.make_type_32.loc27_22 [template = i32]
// CHECK:STDOUT:   %.loc27_22.2: type = converted %int.make_type_32.loc27_22, %.loc27_22.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc36_8.1: type = value_of_initializer %int.make_type_32.loc36 [template = i32]
// CHECK:STDOUT:   %.loc36_8.2: type = converted %int.make_type_32.loc36, %.loc36_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc45: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc45_13.1: type = value_of_initializer %int.make_type_32.loc45 [template = i32]
// CHECK:STDOUT:   %.loc45_13.2: type = converted %int.make_type_32.loc45, %.loc45_13.1 [template = i32]
// CHECK:STDOUT:   %.loc45_16: type = struct_type {.a: i32} [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref %.3 = var x
// CHECK:STDOUT:   %x: ref %.3 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc53_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc53_13.1: type = value_of_initializer %int.make_type_32.loc53_13 [template = i32]
// CHECK:STDOUT:   %.loc53_13.2: type = converted %int.make_type_32.loc53_13, %.loc53_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc53_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc53_22.1: type = value_of_initializer %int.make_type_32.loc53_22 [template = i32]
// CHECK:STDOUT:   %.loc53_22.2: type = converted %int.make_type_32.loc53_22, %.loc53_22.1 [template = i32]
// CHECK:STDOUT:   %.loc53_25: type = struct_type {.b: i32, .c: i32} [template = constants.%.5]
// CHECK:STDOUT:   %y.var: ref %.5 = var y
// CHECK:STDOUT:   %y: ref %.5 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc27_36: %.3 = struct_literal (%.loc27_35)
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT:   %.loc36_22: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc36_32: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %w: i32 = bind_name w, <error>
// CHECK:STDOUT:   %.loc45_26: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc45_34: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %.loc53_35: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc53_43: i32 = int_literal 4 [template = constants.%.8]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
