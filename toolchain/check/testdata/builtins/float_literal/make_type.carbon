// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn FloatLiteral() -> type = "float_literal.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
let f: FloatLiteral() = 1.0;
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FloatLiteral.type: type = fn_type @FloatLiteral [concrete]
// CHECK:STDOUT:   %FloatLiteral: %FloatLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dab: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteral: %FloatLiteral.type = import_ref Main//types, FloatLiteral, loaded [concrete = constants.%FloatLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.dab = binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_21.1: type = splice_block %.loc7_21.3 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %FloatLiteral.ref: %FloatLiteral.type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:     %FloatLiteral.call: init type = call %FloatLiteral.ref() [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc7_21.2: type = value_of_initializer %FloatLiteral.call [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc7_21.3: type = converted %FloatLiteral.call, %.loc7_21.2 [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: Core.FloatLiteral = bind_name f, @__global_init.%float
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
