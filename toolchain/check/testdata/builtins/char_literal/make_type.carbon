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

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
let ascii_x: CharLiteral = 'x';
let not_7_bit: CharLiteral = '\u{80}';
let not_8_bit: CharLiteral = '\u{1E15}';
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %.4ac: Core.CharLiteral = char_value U+0078 [concrete]
// CHECK:STDOUT:   %.ae7: Core.CharLiteral = char_value U+0080 [concrete]
// CHECK:STDOUT:   %.a81: Core.CharLiteral = char_value U+1E15 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CharLiteral: type = import_ref Main//types, CharLiteral, loaded [concrete = Core.CharLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ascii_x.patt: %pattern_type = value_binding_pattern ascii_x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc7: type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %ascii_x: Core.CharLiteral = value_binding ascii_x, @__global_init.%.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_7_bit.patt: %pattern_type = value_binding_pattern not_7_bit [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc8: type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %not_7_bit: Core.CharLiteral = value_binding not_7_bit, @__global_init.%.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_8_bit.patt: %pattern_type = value_binding_pattern not_8_bit [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc9: type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %not_8_bit: Core.CharLiteral = value_binding not_8_bit, @__global_init.%.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: Core.CharLiteral = char_value U+0078 [concrete = constants.%.4ac]
// CHECK:STDOUT:   %.loc8: Core.CharLiteral = char_value U+0080 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %.loc9: Core.CharLiteral = char_value U+1E15 [concrete = constants.%.a81]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
