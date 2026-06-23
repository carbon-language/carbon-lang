// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/greater.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/greater.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();

fn Greater(a: CharLiteral, b: CharLiteral) -> bool = "char_literal.greater";

//@dump-sem-ir-begin
let a: bool = Greater('b', 'a');
let b: bool = Greater('a', 'b');
//@dump-sem-ir-end

// CHECK:STDOUT: --- greater.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.567: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %b.patt.438: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete = constants.%a.patt.567]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_31.1: bool = value_of_initializer @__global_init.%Greater.call.loc21 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc21_31.2: bool = converted @__global_init.%Greater.call.loc21, %.loc21_31.1 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc21_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a: bool = wrapper_binding a, %.loc21_31.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt.438]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_31.1: bool = value_of_initializer @__global_init.%Greater.call.loc22 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc22_31.2: bool = converted @__global_init.%Greater.call.loc22, %.loc22_31.1 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc22_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc22_31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Greater.ref.loc21: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %.loc21_23: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc21_28: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %Greater.call.loc21: init bool = call %Greater.ref.loc21(%.loc21_23, %.loc21_28) [concrete = constants.%true]
// CHECK:STDOUT:   %Greater.ref.loc22: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %.loc22_23: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc22_28: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %Greater.call.loc22: init bool = call %Greater.ref.loc22(%.loc22_23, %.loc22_28) [concrete = constants.%false]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
