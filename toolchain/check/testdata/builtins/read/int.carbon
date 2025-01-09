// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/read/int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/read/int.carbon

import Core library "io";

fn ReadChar() -> i32 = "read.char";

fn Main() {
  let n: i32 = ReadChar();
  let m: i32 = Core.ReadChar();
}

// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %ReadChar.type.fa8: type = fn_type @ReadChar.1 [template]
// CHECK:STDOUT:   %ReadChar.7f4: %ReadChar.type.fa8 = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %ReadChar.type.9f3: type = fn_type @ReadChar.2 [template]
// CHECK:STDOUT:   %ReadChar.01f: %ReadChar.type.9f3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ReadChar = %import_ref.12c
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.12c: %ReadChar.type.9f3 = import_ref Core//io, ReadChar, loaded [template = constants.%ReadChar.01f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ReadChar = %ReadChar.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ReadChar.decl: %ReadChar.type.fa8 = fn_decl @ReadChar.1 [template = constants.%ReadChar.7f4] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReadChar.1() -> %i32 = "read.char";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReadChar.ref.loc16: %ReadChar.type.fa8 = name_ref ReadChar, file.%ReadChar.decl [template = constants.%ReadChar.7f4]
// CHECK:STDOUT:   %read.char.loc16: init %i32 = call %ReadChar.ref.loc16()
// CHECK:STDOUT:   %.loc16_26.1: %i32 = value_of_initializer %read.char.loc16
// CHECK:STDOUT:   %.loc16_26.2: %i32 = converted %read.char.loc16, %.loc16_26.1
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc16_26.2
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %ReadChar.ref.loc17: %ReadChar.type.9f3 = name_ref ReadChar, imports.%import_ref.12c [template = constants.%ReadChar.01f]
// CHECK:STDOUT:   %read.char.loc17: init %i32 = call %ReadChar.ref.loc17()
// CHECK:STDOUT:   %.loc17_31.1: %i32 = value_of_initializer %read.char.loc17
// CHECK:STDOUT:   %.loc17_31.2: %i32 = converted %read.char.loc17, %.loc17_31.1
// CHECK:STDOUT:   %m: %i32 = bind_name m, %.loc17_31.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
