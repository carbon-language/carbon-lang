// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/sub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/sub.carbon

library "[[@TEST_NAME]]";

fn Sub(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.sub";

//@dump-sem-ir-begin
let v1: Core.FloatLiteral = Sub(2.0, 1.0);
let v2: Core.FloatLiteral = Sub(0x1.8p0, 2.5);
let v3: Core.FloatLiteral = Sub(2.5, 0x1.8p0);
let v4: Core.FloatLiteral = Sub(1.0e20, 1.0e-20);
let v5: Core.FloatLiteral = Sub(1.0e-20, 1.0e20);
let v6: Core.FloatLiteral = Sub(1.0, 1.000000000000000000000000000000);
let v7: Core.FloatLiteral = Sub(1.000000000000000000000000000000, 1.0);
let v8: Core.FloatLiteral = Sub(Sub(0.0, 0.9223372036854775807), 0.0000000000000000002);
//@dump-sem-ir-end

// CHECK:STDOUT: --- sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [concrete]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [concrete]
// CHECK:STDOUT:   %v1.patt: %pattern_type = value_binding_pattern v1 [concrete]
// CHECK:STDOUT:   %float.cce: Core.FloatLiteral = float_literal_value 20e-1 [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.6daae3.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %v2.patt: %pattern_type = value_binding_pattern v2 [concrete]
// CHECK:STDOUT:   %float.99ca5c.1: Core.FloatLiteral = float_literal_value 24p-4 [concrete]
// CHECK:STDOUT:   %float.27d350.1: Core.FloatLiteral = float_literal_value 25e-1 [concrete]
// CHECK:STDOUT:   %float.6fa: Core.FloatLiteral = float_literal_value -10000e-4 [concrete]
// CHECK:STDOUT:   %v3.patt: %pattern_type = value_binding_pattern v3 [concrete]
// CHECK:STDOUT:   %float.27d350.2: Core.FloatLiteral = float_literal_value 25e-1 [concrete]
// CHECK:STDOUT:   %float.99ca5c.2: Core.FloatLiteral = float_literal_value 24p-4 [concrete]
// CHECK:STDOUT:   %float.1a9: Core.FloatLiteral = float_literal_value 10000e-4 [concrete]
// CHECK:STDOUT:   %v4.patt: %pattern_type = value_binding_pattern v4 [concrete]
// CHECK:STDOUT:   %float.e08d73.1: Core.FloatLiteral = float_literal_value 10e19 [concrete]
// CHECK:STDOUT:   %float.20e171.1: Core.FloatLiteral = float_literal_value 10e-21 [concrete]
// CHECK:STDOUT:   %float.57a: Core.FloatLiteral = float_literal_value 99999999999999999999999999999999999999990e-21 [concrete]
// CHECK:STDOUT:   %v5.patt: %pattern_type = value_binding_pattern v5 [concrete]
// CHECK:STDOUT:   %float.20e171.2: Core.FloatLiteral = float_literal_value 10e-21 [concrete]
// CHECK:STDOUT:   %float.e08d73.2: Core.FloatLiteral = float_literal_value 10e19 [concrete]
// CHECK:STDOUT:   %float.4fd: Core.FloatLiteral = float_literal_value -99999999999999999999999999999999999999990e-21 [concrete]
// CHECK:STDOUT:   %v6.patt: %pattern_type = value_binding_pattern v6 [concrete]
// CHECK:STDOUT:   %float.6daae3.3: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.54c8cd.1: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete]
// CHECK:STDOUT:   %float.d00f07.1: Core.FloatLiteral = float_literal_value 0e-30 [concrete]
// CHECK:STDOUT:   %v7.patt: %pattern_type = value_binding_pattern v7 [concrete]
// CHECK:STDOUT:   %float.54c8cd.2: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete]
// CHECK:STDOUT:   %float.6daae3.4: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.d00f07.2: Core.FloatLiteral = float_literal_value 0e-30 [concrete]
// CHECK:STDOUT:   %v8.patt: %pattern_type = value_binding_pattern v8 [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.09a: Core.FloatLiteral = float_literal_value 9223372036854775807e-19 [concrete]
// CHECK:STDOUT:   %float.18e: Core.FloatLiteral = float_literal_value -9223372036854775807e-19 [concrete]
// CHECK:STDOUT:   %float.382: Core.FloatLiteral = float_literal_value 2e-19 [concrete]
// CHECK:STDOUT:   %float.e21: Core.FloatLiteral = float_literal_value 9223372036854775807e-19 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .FloatLiteral = %Core.FloatLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.FloatLiteral: type = import_ref Core//prelude/parts/float_literal, FloatLiteral, loaded [concrete = Core.FloatLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc18_41.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc18 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %.loc18_41.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc18, %.loc18_41.1 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %.loc18_13: type = splice_block %FloatLiteral.ref.loc18 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc18: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v1: Core.FloatLiteral = wrapper_binding v1, %.loc18_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v1.patt: %pattern_type = value_binding_pattern v1 [concrete = constants.%v1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_45.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc19 [concrete = constants.%float.6fa]
// CHECK:STDOUT:   %.loc19_45.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc19, %.loc19_45.1 [concrete = constants.%float.6fa]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %FloatLiteral.ref.loc19 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc19: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: Core.FloatLiteral = wrapper_binding v2, %.loc19_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %pattern_type = value_binding_pattern v2 [concrete = constants.%v2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_45.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc20 [concrete = constants.%float.1a9]
// CHECK:STDOUT:   %.loc20_45.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc20, %.loc20_45.1 [concrete = constants.%float.1a9]
// CHECK:STDOUT:   %.loc20_13: type = splice_block %FloatLiteral.ref.loc20 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc20: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3: Core.FloatLiteral = wrapper_binding v3, %.loc20_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v3.patt: %pattern_type = value_binding_pattern v3 [concrete = constants.%v3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_48.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc21 [concrete = constants.%float.57a]
// CHECK:STDOUT:   %.loc21_48.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc21, %.loc21_48.1 [concrete = constants.%float.57a]
// CHECK:STDOUT:   %.loc21_13: type = splice_block %FloatLiteral.ref.loc21 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc21: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc21: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4: Core.FloatLiteral = wrapper_binding v4, %.loc21_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v4.patt: %pattern_type = value_binding_pattern v4 [concrete = constants.%v4.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_48.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc22 [concrete = constants.%float.4fd]
// CHECK:STDOUT:   %.loc22_48.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc22, %.loc22_48.1 [concrete = constants.%float.4fd]
// CHECK:STDOUT:   %.loc22_13: type = splice_block %FloatLiteral.ref.loc22 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc22: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5: Core.FloatLiteral = wrapper_binding v5, %.loc22_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v5.patt: %pattern_type = value_binding_pattern v5 [concrete = constants.%v5.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_70.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc23 [concrete = constants.%float.d00f07.1]
// CHECK:STDOUT:   %.loc23_70.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc23, %.loc23_70.1 [concrete = constants.%float.d00f07.1]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %FloatLiteral.ref.loc23 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc23: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v6: Core.FloatLiteral = wrapper_binding v6, %.loc23_70.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v6.patt: %pattern_type = value_binding_pattern v6 [concrete = constants.%v6.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_70.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc24 [concrete = constants.%float.d00f07.2]
// CHECK:STDOUT:   %.loc24_70.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc24, %.loc24_70.1 [concrete = constants.%float.d00f07.2]
// CHECK:STDOUT:   %.loc24_13: type = splice_block %FloatLiteral.ref.loc24 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc24: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v7: Core.FloatLiteral = wrapper_binding v7, %.loc24_70.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v7.patt: %pattern_type = value_binding_pattern v7 [concrete = constants.%v7.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_87.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc25_87 [concrete = constants.%float.e21]
// CHECK:STDOUT:   %.loc25_87.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc25_87, %.loc25_87.1 [concrete = constants.%float.e21]
// CHECK:STDOUT:   %.loc25_13: type = splice_block %FloatLiteral.ref.loc25 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc25: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc25: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v8: Core.FloatLiteral = wrapper_binding v8, %.loc25_87.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v8.patt: %pattern_type = value_binding_pattern v8 [concrete = constants.%v8.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc18: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc18_33: Core.FloatLiteral = float_literal_value 20e-1 [concrete = constants.%float.cce]
// CHECK:STDOUT:   %float.loc18_38: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %Sub.call.loc18: init Core.FloatLiteral = call %Sub.ref.loc18(%float.loc18_33, %float.loc18_38) [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %Sub.ref.loc19: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc19_33: Core.FloatLiteral = float_literal_value 24p-4 [concrete = constants.%float.99ca5c.1]
// CHECK:STDOUT:   %float.loc19_42: Core.FloatLiteral = float_literal_value 25e-1 [concrete = constants.%float.27d350.1]
// CHECK:STDOUT:   %Sub.call.loc19: init Core.FloatLiteral = call %Sub.ref.loc19(%float.loc19_33, %float.loc19_42) [concrete = constants.%float.6fa]
// CHECK:STDOUT:   %Sub.ref.loc20: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc20_33: Core.FloatLiteral = float_literal_value 25e-1 [concrete = constants.%float.27d350.2]
// CHECK:STDOUT:   %float.loc20_38: Core.FloatLiteral = float_literal_value 24p-4 [concrete = constants.%float.99ca5c.2]
// CHECK:STDOUT:   %Sub.call.loc20: init Core.FloatLiteral = call %Sub.ref.loc20(%float.loc20_33, %float.loc20_38) [concrete = constants.%float.1a9]
// CHECK:STDOUT:   %Sub.ref.loc21: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc21_33: Core.FloatLiteral = float_literal_value 10e19 [concrete = constants.%float.e08d73.1]
// CHECK:STDOUT:   %float.loc21_41: Core.FloatLiteral = float_literal_value 10e-21 [concrete = constants.%float.20e171.1]
// CHECK:STDOUT:   %Sub.call.loc21: init Core.FloatLiteral = call %Sub.ref.loc21(%float.loc21_33, %float.loc21_41) [concrete = constants.%float.57a]
// CHECK:STDOUT:   %Sub.ref.loc22: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc22_33: Core.FloatLiteral = float_literal_value 10e-21 [concrete = constants.%float.20e171.2]
// CHECK:STDOUT:   %float.loc22_42: Core.FloatLiteral = float_literal_value 10e19 [concrete = constants.%float.e08d73.2]
// CHECK:STDOUT:   %Sub.call.loc22: init Core.FloatLiteral = call %Sub.ref.loc22(%float.loc22_33, %float.loc22_42) [concrete = constants.%float.4fd]
// CHECK:STDOUT:   %Sub.ref.loc23: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc23_33: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.3]
// CHECK:STDOUT:   %float.loc23_38: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete = constants.%float.54c8cd.1]
// CHECK:STDOUT:   %Sub.call.loc23: init Core.FloatLiteral = call %Sub.ref.loc23(%float.loc23_33, %float.loc23_38) [concrete = constants.%float.d00f07.1]
// CHECK:STDOUT:   %Sub.ref.loc24: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc24_33: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete = constants.%float.54c8cd.2]
// CHECK:STDOUT:   %float.loc24_67: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.4]
// CHECK:STDOUT:   %Sub.call.loc24: init Core.FloatLiteral = call %Sub.ref.loc24(%float.loc24_33, %float.loc24_67) [concrete = constants.%float.d00f07.2]
// CHECK:STDOUT:   %Sub.ref.loc25_29: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc25_33: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc25_37: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %float.loc25_42: Core.FloatLiteral = float_literal_value 9223372036854775807e-19 [concrete = constants.%float.09a]
// CHECK:STDOUT:   %Sub.call.loc25_63: init Core.FloatLiteral = call %Sub.ref.loc25_33(%float.loc25_37, %float.loc25_42) [concrete = constants.%float.18e]
// CHECK:STDOUT:   %float.loc25_66: Core.FloatLiteral = float_literal_value 2e-19 [concrete = constants.%float.382]
// CHECK:STDOUT:   %.loc25_63.1: Core.FloatLiteral = value_of_initializer %Sub.call.loc25_63 [concrete = constants.%float.18e]
// CHECK:STDOUT:   %.loc25_63.2: Core.FloatLiteral = converted %Sub.call.loc25_63, %.loc25_63.1 [concrete = constants.%float.18e]
// CHECK:STDOUT:   %Sub.call.loc25_87: init Core.FloatLiteral = call %Sub.ref.loc25_29(%.loc25_63.2, %float.loc25_66) [concrete = constants.%float.e21]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
