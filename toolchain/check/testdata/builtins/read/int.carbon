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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ReadChar.type.fa8: type = fn_type @ReadChar.1 [concrete]
// CHECK:STDOUT:   %ReadChar.7f4: %ReadChar.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReadChar.type.9f3: type = fn_type @ReadChar.2 [concrete]
// CHECK:STDOUT:   %ReadChar.01f: %ReadChar.type.9f3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ReadChar = %Core.ReadChar
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ReadChar: %ReadChar.type.9f3 = import_ref Core//io, ReadChar, loaded [concrete = constants.%ReadChar.01f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ReadChar = %ReadChar.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ReadChar.decl: %ReadChar.type.fa8 = fn_decl @ReadChar.1 [concrete = constants.%ReadChar.7f4] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReadChar.1() -> %i32 = "read.char";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReadChar.ref.loc16: %ReadChar.type.fa8 = name_ref ReadChar, file.%ReadChar.decl [concrete = constants.%ReadChar.7f4]
// CHECK:STDOUT:   %read.char.loc16: init %i32 = call %ReadChar.ref.loc16()
// CHECK:STDOUT:   %.loc16_10: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_25.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc16_25.2: ref %i32 = temporary %.loc16_25.1, %read.char.loc16
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %.loc16_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %ReadChar.ref.loc17: %ReadChar.type.9f3 = name_ref ReadChar, imports.%Core.ReadChar [concrete = constants.%ReadChar.01f]
// CHECK:STDOUT:   %read.char.loc17: init %i32 = call %ReadChar.ref.loc17()
// CHECK:STDOUT:   %.loc17_10: type = splice_block %i32.loc17 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_30.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc17_30.2: ref %i32 = temporary %.loc17_30.1, %read.char.loc17
// CHECK:STDOUT:   %m: ref %i32 = bind_name m, %.loc17_30.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
