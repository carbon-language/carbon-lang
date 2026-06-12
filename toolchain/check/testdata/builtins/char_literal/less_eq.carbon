// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/less_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/less_eq.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();

fn LessEq(a: CharLiteral, b: CharLiteral) -> bool = "char_literal.less_eq";

//@dump-sem-ir-begin
let a: bool = LessEq('a', 'a');
let b: bool = LessEq('a', 'b');
let c: bool = LessEq('b', 'a');
//@dump-sem-ir-end

// CHECK:STDOUT: --- less_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [concrete]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc21_30.1: bool = value_of_initializer @__global_init.%LessEq.call.loc21 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc21_30.2: bool = converted @__global_init.%LessEq.call.loc21, %.loc21_30.1 [concrete = constants.%true]
// CHECK:STDOUT:   %a: bool = value_binding a, %.loc21_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc22_30.1: bool = value_of_initializer @__global_init.%LessEq.call.loc22 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc22_30.2: bool = converted @__global_init.%LessEq.call.loc22, %.loc22_30.1 [concrete = constants.%true]
// CHECK:STDOUT:   %b: bool = value_binding b, %.loc22_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc23_30.1: bool = value_of_initializer @__global_init.%LessEq.call.loc23 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc23_30.2: bool = converted @__global_init.%LessEq.call.loc23, %.loc23_30.1 [concrete = constants.%false]
// CHECK:STDOUT:   %c: bool = value_binding c, %.loc23_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LessEq.ref.loc21: %LessEq.type = name_ref LessEq, file.%LessEq.decl [concrete = constants.%LessEq]
// CHECK:STDOUT:   %.loc21_22: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc21_27: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %LessEq.call.loc21: init bool = call %LessEq.ref.loc21(%.loc21_22, %.loc21_27) [concrete = constants.%true]
// CHECK:STDOUT:   %LessEq.ref.loc22: %LessEq.type = name_ref LessEq, file.%LessEq.decl [concrete = constants.%LessEq]
// CHECK:STDOUT:   %.loc22_22: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc22_27: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %LessEq.call.loc22: init bool = call %LessEq.ref.loc22(%.loc22_22, %.loc22_27) [concrete = constants.%true]
// CHECK:STDOUT:   %LessEq.ref.loc23: %LessEq.type = name_ref LessEq, file.%LessEq.decl [concrete = constants.%LessEq]
// CHECK:STDOUT:   %.loc23_22: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc23_27: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %LessEq.call.loc23: init bool = call %LessEq.ref.loc23(%.loc23_22, %.loc23_27) [concrete = constants.%false]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
