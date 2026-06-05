// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/less.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();

fn Less(a: CharLiteral, b: CharLiteral) -> bool = "char_literal.less";

//@dump-sem-ir-begin
let a: bool = Less('a', 'b');
let b: bool = Less('b', 'a');
//@dump-sem-ir-end

// CHECK:STDOUT: --- less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc21_28.1: bool = value_of_initializer @__global_init.%Less.call.loc21 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc21_28.2: bool = converted @__global_init.%Less.call.loc21, %.loc21_28.1 [concrete = constants.%true]
// CHECK:STDOUT:   %a: bool = value_binding a, %.loc21_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc22_28.1: bool = value_of_initializer @__global_init.%Less.call.loc22 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc22_28.2: bool = converted @__global_init.%Less.call.loc22, %.loc22_28.1 [concrete = constants.%false]
// CHECK:STDOUT:   %b: bool = value_binding b, %.loc22_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref.loc21: %Less.type = name_ref Less, file.%Less.decl [concrete = constants.%Less]
// CHECK:STDOUT:   %.loc21_20: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc21_25: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %Less.call.loc21: init bool = call %Less.ref.loc21(%.loc21_20, %.loc21_25) [concrete = constants.%true]
// CHECK:STDOUT:   %Less.ref.loc22: %Less.type = name_ref Less, file.%Less.decl [concrete = constants.%Less]
// CHECK:STDOUT:   %.loc22_20: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc22_25: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %Less.call.loc22: init bool = call %Less.ref.loc22(%.loc22_20, %.loc22_25) [concrete = constants.%false]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
