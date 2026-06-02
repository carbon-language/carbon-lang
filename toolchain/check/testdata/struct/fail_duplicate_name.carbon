// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+7]]:39: error: duplicated field name `b` in struct literal [StructNameDuplicate]
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                                       ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+4]]:31: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                               ^
// CHECK:STDERR:
var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};

// CHECK:STDOUT: --- fail_duplicate_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.a6c = struct_value (%int_1) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.72d: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.79c: type = pattern_type %struct_type.a.72d [concrete]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.e2e: type = pattern_type %struct_type.b.c [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern <error>, <error> [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %return: <error> = return_slot <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %i32.loc31_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc31_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: <error> = value_binding v, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.c6b = value_binding_pattern w [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.loc40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %w: %i32 = value_binding w, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.79c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.79c = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.72d = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc49: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.72d] {
// CHECK:STDOUT:     %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.72d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.72d = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.e2e = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.e2e = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.b.c = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc58: type = splice_block %struct_type.b.c [concrete = constants.%struct_type.b.c] {
// CHECK:STDOUT:     %i32.loc58_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc58_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %i32} [concrete = constants.%struct_type.b.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.b.c = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc31: %struct_type.a.a6c = struct_literal (%int_1.loc31) [concrete = constants.%struct]
// CHECK:STDOUT:   %int_1.loc40: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc40: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %def.ref: <error> = name_ref def, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc49: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
