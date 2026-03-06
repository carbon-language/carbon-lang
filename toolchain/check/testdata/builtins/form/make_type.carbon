// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/form/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/form/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn Form() -> type = "form.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
let f: Form() = form(var ());
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Form.type: type = fn_type @Form [concrete]
// CHECK:STDOUT:   %Form: %Form.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Form: %Form.type = import_ref Main//types, Form, loaded [concrete = constants.%Form]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.13f = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_13.1: type = splice_block %.loc7_13.3 [concrete = Core.Form] {
// CHECK:STDOUT:     %Form.ref: %Form.type = name_ref Form, imports.%Main.Form [concrete = constants.%Form]
// CHECK:STDOUT:     %Form.call: init type = call %Form.ref() [concrete = Core.Form]
// CHECK:STDOUT:     %.loc7_13.2: type = value_of_initializer %Form.call [concrete = Core.Form]
// CHECK:STDOUT:     %.loc7_13.3: type = converted %Form.call, %.loc7_13.2 [concrete = Core.Form]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: Core.Form = value_binding f, @__global_init.%.loc7_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.2: type = converted %.loc7_27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc7_22: Core.Form = init_form %.loc7_27.2 [concrete = constants.%.262]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
