// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn CharLiteral() -> type = "char_literal.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
let ascii_x: CharLiteral() = 'x';
let not_7_bit: CharLiteral() = '\u{80}';
let not_8_bit: CharLiteral() = '\u{1E15}';
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CharLiteral.type: type = fn_type @CharLiteral [concrete]
// CHECK:STDOUT:   %CharLiteral: %CharLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %.4ac: Core.CharLiteral = char_value U+0078 [concrete]
// CHECK:STDOUT:   %.ae7: Core.CharLiteral = char_value U+0080 [concrete]
// CHECK:STDOUT:   %.a81: Core.CharLiteral = char_value U+1E15 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CharLiteral: %CharLiteral.type = import_ref Main//types, CharLiteral, loaded [concrete = constants.%CharLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ascii_x.patt: %pattern_type.8c6 = value_binding_pattern ascii_x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_26.1: type = splice_block %.loc26_26.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc26: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc26: init type = call %CharLiteral.ref.loc26() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc26_26.2: type = value_of_initializer %CharLiteral.call.loc26 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc26_26.3: type = converted %CharLiteral.call.loc26, %.loc26_26.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ascii_x: Core.CharLiteral = value_binding ascii_x, @__global_init.%.loc26
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_7_bit.patt: %pattern_type.8c6 = value_binding_pattern not_7_bit [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_28.1: type = splice_block %.loc27_28.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc27: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc27: init type = call %CharLiteral.ref.loc27() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc27_28.2: type = value_of_initializer %CharLiteral.call.loc27 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc27_28.3: type = converted %CharLiteral.call.loc27, %.loc27_28.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_7_bit: Core.CharLiteral = value_binding not_7_bit, @__global_init.%.loc27
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_8_bit.patt: %pattern_type.8c6 = value_binding_pattern not_8_bit [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_28.1: type = splice_block %.loc28_28.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc28: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc28: init type = call %CharLiteral.ref.loc28() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc28_28.2: type = value_of_initializer %CharLiteral.call.loc28 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc28_28.3: type = converted %CharLiteral.call.loc28, %.loc28_28.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_8_bit: Core.CharLiteral = value_binding not_8_bit, @__global_init.%.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26: Core.CharLiteral = char_value U+0078 [concrete = constants.%.4ac]
// CHECK:STDOUT:   %.loc27: Core.CharLiteral = char_value U+0080 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %.loc28: Core.CharLiteral = char_value U+1E15 [concrete = constants.%.a81]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
