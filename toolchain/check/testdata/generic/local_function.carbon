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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/local_function.carbon

fn Test[unused T:! type]() {
  fn Local() {
  }
  Local();
}

// CHECK:STDOUT: --- local_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %Local.type: type = fn_type @Local, @Test(%T) [symbolic]
// CHECK:STDOUT:   %Local: %Local.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Local.specific_fn: <specific function> = specific_function %Local, @Local(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %T.patt.loc15_17.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20.1: type = splice_block %.loc15_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc15_17.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_17.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Local.type: type = fn_type @Local, @Test(%T.loc15_17.1) [symbolic = %Local.type (constants.%Local.type)]
// CHECK:STDOUT:   %Local: @Test.%Local.type (%Local.type) = struct_value () [symbolic = %Local (constants.%Local)]
// CHECK:STDOUT:   %Local.specific_fn.loc18_3.2: <specific function> = specific_function %Local, @Local(%T.loc15_17.1) [symbolic = %Local.specific_fn.loc18_3.2 (constants.%Local.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Local.decl: @Test.%Local.type (%Local.type) = fn_decl @Local [symbolic = @Test.%Local (constants.%Local)] {} {}
// CHECK:STDOUT:     %Local.ref: @Test.%Local.type (%Local.type) = name_ref Local, %Local.decl [symbolic = %Local (constants.%Local)]
// CHECK:STDOUT:     %Local.specific_fn.loc18_3.1: <specific function> = specific_function %Local.ref, @Local(constants.%T) [symbolic = %Local.specific_fn.loc18_3.2 (constants.%Local.specific_fn)]
// CHECK:STDOUT:     %Local.call: init %empty_tuple.type = call %Local.specific_fn.loc18_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Local(@Test.%T.loc15_17.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_17.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
