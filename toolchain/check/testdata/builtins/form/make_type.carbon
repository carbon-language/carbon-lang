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

// TODO: test more more realistic usages once they're supported.

//@dump-sem-ir-begin
var f: Form();
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Form.type: type = fn_type @Form [concrete]
// CHECK:STDOUT:   %Form: %Form.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Form: %Form.type = import_ref Main//types, Form, loaded [concrete = constants.%Form]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.13f = ref_binding_pattern f [concrete]
// CHECK:STDOUT:     %f.var_patt: %pattern_type.13f = var_pattern %f.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref Core.Form = var %f.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_13.1: type = splice_block %.loc9_13.3 [concrete = Core.Form] {
// CHECK:STDOUT:     %Form.ref: %Form.type = name_ref Form, imports.%Main.Form [concrete = constants.%Form]
// CHECK:STDOUT:     %Form.call: init type = call %Form.ref() [concrete = Core.Form]
// CHECK:STDOUT:     %.loc9_13.2: type = value_of_initializer %Form.call [concrete = Core.Form]
// CHECK:STDOUT:     %.loc9_13.3: type = converted %Form.call, %.loc9_13.2 [concrete = Core.Form]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref Core.Form = ref_binding f, %f.var [concrete = %f.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
