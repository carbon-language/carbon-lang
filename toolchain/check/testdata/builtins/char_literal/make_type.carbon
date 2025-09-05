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
var ascii_x: CharLiteral() = 'x';
var not_7_bit: CharLiteral() = '\u{80}';
var not_8_bit: CharLiteral() = '\u{1E15}';
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
// CHECK:STDOUT:     %ascii_x.patt: %pattern_type.8c6 = binding_pattern ascii_x [concrete]
// CHECK:STDOUT:     %ascii_x.var_patt: %pattern_type.8c6 = var_pattern %ascii_x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ascii_x.var: ref Core.CharLiteral = var %ascii_x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_26.1: type = splice_block %.loc7_26.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc7: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc7: init type = call %CharLiteral.ref.loc7() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc7_26.2: type = value_of_initializer %CharLiteral.call.loc7 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc7_26.3: type = converted %CharLiteral.call.loc7, %.loc7_26.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ascii_x: ref Core.CharLiteral = bind_name ascii_x, %ascii_x.var [concrete = %ascii_x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_7_bit.patt: %pattern_type.8c6 = binding_pattern not_7_bit [concrete]
// CHECK:STDOUT:     %not_7_bit.var_patt: %pattern_type.8c6 = var_pattern %not_7_bit.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_7_bit.var: ref Core.CharLiteral = var %not_7_bit.var_patt [concrete]
// CHECK:STDOUT:   %.loc8_28.1: type = splice_block %.loc8_28.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc8: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc8: init type = call %CharLiteral.ref.loc8() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc8_28.2: type = value_of_initializer %CharLiteral.call.loc8 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc8_28.3: type = converted %CharLiteral.call.loc8, %.loc8_28.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_7_bit: ref Core.CharLiteral = bind_name not_7_bit, %not_7_bit.var [concrete = %not_7_bit.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_8_bit.patt: %pattern_type.8c6 = binding_pattern not_8_bit [concrete]
// CHECK:STDOUT:     %not_8_bit.var_patt: %pattern_type.8c6 = var_pattern %not_8_bit.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_8_bit.var: ref Core.CharLiteral = var %not_8_bit.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_28.1: type = splice_block %.loc9_28.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %CharLiteral.ref.loc9: %CharLiteral.type = name_ref CharLiteral, imports.%Main.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call.loc9: init type = call %CharLiteral.ref.loc9() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc9_28.2: type = value_of_initializer %CharLiteral.call.loc9 [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc9_28.3: type = converted %CharLiteral.call.loc9, %.loc9_28.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_8_bit: ref Core.CharLiteral = bind_name not_8_bit, %not_8_bit.var [concrete = %not_8_bit.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: Core.CharLiteral = char_value U+0078 [concrete = constants.%.4ac]
// CHECK:STDOUT:   assign file.%ascii_x.var, %.loc7
// CHECK:STDOUT:   %.loc8: Core.CharLiteral = char_value U+0080 [concrete = constants.%.ae7]
// CHECK:STDOUT:   assign file.%not_7_bit.var, %.loc8
// CHECK:STDOUT:   %.loc9: Core.CharLiteral = char_value U+1E15 [concrete = constants.%.a81]
// CHECK:STDOUT:   assign file.%not_8_bit.var, %.loc9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
