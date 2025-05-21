// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/read/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/read/char.carbon

// --- char.carbon

import Core library "io";

fn ReadChar() -> i32 = "read.char";

fn Main() {
  //@dump-sem-ir-begin
  let n: i32 = ReadChar();
  let m: i32 = Core.ReadChar();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ReadChar.type.fa8: type = fn_type @ReadChar.1 [concrete]
// CHECK:STDOUT:   %ReadChar.7f4: %ReadChar.type.fa8 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ReadChar: %ReadChar.type.9f3 = import_ref Core//io, ReadChar, loaded [concrete = constants.%ReadChar.01f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReadChar.ref.loc8: %ReadChar.type.fa8 = name_ref ReadChar, file.%ReadChar.decl [concrete = constants.%ReadChar.7f4]
// CHECK:STDOUT:   %read.char.loc8: init %i32 = call %ReadChar.ref.loc8()
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_25.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc8_25.2: ref %i32 = temporary %.loc8_25.1, %read.char.loc8
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %.loc8_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = binding_pattern m [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %ReadChar.ref.loc9: %ReadChar.type.9f3 = name_ref ReadChar, imports.%Core.ReadChar [concrete = constants.%ReadChar.01f]
// CHECK:STDOUT:   %read.char.loc9: init %i32 = call %ReadChar.ref.loc9()
// CHECK:STDOUT:   %.loc9_10: type = splice_block %i32.loc9 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_30.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc9_30.2: ref %i32 = temporary %.loc9_30.1, %read.char.loc9
// CHECK:STDOUT:   %m: ref %i32 = bind_name m, %.loc9_30.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
