// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/add.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.add";

//@dump-sem-ir-begin
let v1: Core.FloatLiteral = Add(1.0, 1.0);
let v2: Core.FloatLiteral = Add(0x1.0p0, 1.0);
let v3: Core.FloatLiteral = Add(0x1.8p0, 2.5);
let v4: Core.FloatLiteral = Add(1.0e20, 1.0e-20);
let v5: Core.FloatLiteral = Add(1.0e-20, 1.0e20);
let v6: Core.FloatLiteral = Add(1.0, 1.000000000000000000000000000000);
let v7: Core.FloatLiteral = Add(0.9223372036854775807, 0.0000000000000000001);
// CHECK:STDERR: add.carbon:[[@LINE+4]]:29: error: conversion of binary float literal 16*2^1023999999996 to decimal would exceed the maximum supported integer width of 8388608 [CompileTimeDecadicFloatLiteralTooLarge]
// CHECK:STDERR: let v8: Core.FloatLiteral = Add(0x1.0p1024000000000, 0.0000000000000000001);
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let v8: Core.FloatLiteral = Add(0x1.0p1024000000000, 0.0000000000000000001);
//@dump-sem-ir-end

// CHECK:STDOUT: --- add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [concrete]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [concrete]
// CHECK:STDOUT:   %v1.patt: %pattern_type = value_binding_pattern v1 [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.6daae3.2: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.cce: Core.FloatLiteral = float_literal_value 20e-1 [concrete]
// CHECK:STDOUT:   %v2.patt: %pattern_type = value_binding_pattern v2 [concrete]
// CHECK:STDOUT:   %float.8b7: Core.FloatLiteral = float_literal_value 16p-4 [concrete]
// CHECK:STDOUT:   %float.6daae3.3: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.074: Core.FloatLiteral = float_literal_value 20000e-4 [concrete]
// CHECK:STDOUT:   %v3.patt: %pattern_type = value_binding_pattern v3 [concrete]
// CHECK:STDOUT:   %float.99c: Core.FloatLiteral = float_literal_value 24p-4 [concrete]
// CHECK:STDOUT:   %float.27d: Core.FloatLiteral = float_literal_value 25e-1 [concrete]
// CHECK:STDOUT:   %float.132: Core.FloatLiteral = float_literal_value 40000e-4 [concrete]
// CHECK:STDOUT:   %v4.patt: %pattern_type = value_binding_pattern v4 [concrete]
// CHECK:STDOUT:   %float.e08d73.1: Core.FloatLiteral = float_literal_value 10e19 [concrete]
// CHECK:STDOUT:   %float.20e171.1: Core.FloatLiteral = float_literal_value 10e-21 [concrete]
// CHECK:STDOUT:   %float.ed50b1.1: Core.FloatLiteral = float_literal_value 100000000000000000000000000000000000000010e-21 [concrete]
// CHECK:STDOUT:   %v5.patt: %pattern_type = value_binding_pattern v5 [concrete]
// CHECK:STDOUT:   %float.20e171.2: Core.FloatLiteral = float_literal_value 10e-21 [concrete]
// CHECK:STDOUT:   %float.e08d73.2: Core.FloatLiteral = float_literal_value 10e19 [concrete]
// CHECK:STDOUT:   %float.ed50b1.2: Core.FloatLiteral = float_literal_value 100000000000000000000000000000000000000010e-21 [concrete]
// CHECK:STDOUT:   %v6.patt: %pattern_type = value_binding_pattern v6 [concrete]
// CHECK:STDOUT:   %float.6daae3.4: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.54c: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete]
// CHECK:STDOUT:   %float.fc1: Core.FloatLiteral = float_literal_value 2000000000000000000000000000000e-30 [concrete]
// CHECK:STDOUT:   %v7.patt: %pattern_type = value_binding_pattern v7 [concrete]
// CHECK:STDOUT:   %float.09a: Core.FloatLiteral = float_literal_value 9223372036854775807e-19 [concrete]
// CHECK:STDOUT:   %float.fbf8ca.1: Core.FloatLiteral = float_literal_value 1e-19 [concrete]
// CHECK:STDOUT:   %float.36a: Core.FloatLiteral = float_literal_value 9223372036854775808e-19 [concrete]
// CHECK:STDOUT:   %v8.patt: %pattern_type = value_binding_pattern v8 [concrete]
// CHECK:STDOUT:   %float.b31: Core.FloatLiteral = float_literal_value 16p1023999999996 [concrete]
// CHECK:STDOUT:   %float.fbf8ca.2: Core.FloatLiteral = float_literal_value 1e-19 [concrete]
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
// CHECK:STDOUT:   %.loc20_41.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc20 [concrete = constants.%float.cce]
// CHECK:STDOUT:   %.loc20_41.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc20, %.loc20_41.1 [concrete = constants.%float.cce]
// CHECK:STDOUT:   %.loc20_13: type = splice_block %FloatLiteral.ref.loc20 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc20: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v1: Core.FloatLiteral = wrapper_binding v1, %.loc20_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v1.patt: %pattern_type = value_binding_pattern v1 [concrete = constants.%v1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_45.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc21 [concrete = constants.%float.074]
// CHECK:STDOUT:   %.loc21_45.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc21, %.loc21_45.1 [concrete = constants.%float.074]
// CHECK:STDOUT:   %.loc21_13: type = splice_block %FloatLiteral.ref.loc21 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc21: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc21: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: Core.FloatLiteral = wrapper_binding v2, %.loc21_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %pattern_type = value_binding_pattern v2 [concrete = constants.%v2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_45.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc22 [concrete = constants.%float.132]
// CHECK:STDOUT:   %.loc22_45.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc22, %.loc22_45.1 [concrete = constants.%float.132]
// CHECK:STDOUT:   %.loc22_13: type = splice_block %FloatLiteral.ref.loc22 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc22: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3: Core.FloatLiteral = wrapper_binding v3, %.loc22_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v3.patt: %pattern_type = value_binding_pattern v3 [concrete = constants.%v3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_48.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc23 [concrete = constants.%float.ed50b1.1]
// CHECK:STDOUT:   %.loc23_48.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc23, %.loc23_48.1 [concrete = constants.%float.ed50b1.1]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %FloatLiteral.ref.loc23 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc23: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4: Core.FloatLiteral = wrapper_binding v4, %.loc23_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v4.patt: %pattern_type = value_binding_pattern v4 [concrete = constants.%v4.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_48.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc24 [concrete = constants.%float.ed50b1.2]
// CHECK:STDOUT:   %.loc24_48.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc24, %.loc24_48.1 [concrete = constants.%float.ed50b1.2]
// CHECK:STDOUT:   %.loc24_13: type = splice_block %FloatLiteral.ref.loc24 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc24: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5: Core.FloatLiteral = wrapper_binding v5, %.loc24_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v5.patt: %pattern_type = value_binding_pattern v5 [concrete = constants.%v5.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_70.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc25 [concrete = constants.%float.fc1]
// CHECK:STDOUT:   %.loc25_70.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc25, %.loc25_70.1 [concrete = constants.%float.fc1]
// CHECK:STDOUT:   %.loc25_13: type = splice_block %FloatLiteral.ref.loc25 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc25: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc25: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v6: Core.FloatLiteral = wrapper_binding v6, %.loc25_70.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v6.patt: %pattern_type = value_binding_pattern v6 [concrete = constants.%v6.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_77.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc26 [concrete = constants.%float.36a]
// CHECK:STDOUT:   %.loc26_77.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc26, %.loc26_77.1 [concrete = constants.%float.36a]
// CHECK:STDOUT:   %.loc26_13: type = splice_block %FloatLiteral.ref.loc26 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc26: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc26: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v7: Core.FloatLiteral = wrapper_binding v7, %.loc26_77.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v7.patt: %pattern_type = value_binding_pattern v7 [concrete = constants.%v7.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_75.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc27 [concrete = <error>]
// CHECK:STDOUT:   %.loc27_75.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc27, %.loc27_75.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc27_13: type = splice_block %FloatLiteral.ref.loc27 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc27: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v8: Core.FloatLiteral = wrapper_binding v8, %.loc27_75.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v8.patt: %pattern_type = value_binding_pattern v8 [concrete = constants.%v8.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref.loc20: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc20_33: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %float.loc20_38: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.2]
// CHECK:STDOUT:   %Add.call.loc20: init Core.FloatLiteral = call %Add.ref.loc20(%float.loc20_33, %float.loc20_38) [concrete = constants.%float.cce]
// CHECK:STDOUT:   %Add.ref.loc21: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc21_33: Core.FloatLiteral = float_literal_value 16p-4 [concrete = constants.%float.8b7]
// CHECK:STDOUT:   %float.loc21_42: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.3]
// CHECK:STDOUT:   %Add.call.loc21: init Core.FloatLiteral = call %Add.ref.loc21(%float.loc21_33, %float.loc21_42) [concrete = constants.%float.074]
// CHECK:STDOUT:   %Add.ref.loc22: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc22_33: Core.FloatLiteral = float_literal_value 24p-4 [concrete = constants.%float.99c]
// CHECK:STDOUT:   %float.loc22_42: Core.FloatLiteral = float_literal_value 25e-1 [concrete = constants.%float.27d]
// CHECK:STDOUT:   %Add.call.loc22: init Core.FloatLiteral = call %Add.ref.loc22(%float.loc22_33, %float.loc22_42) [concrete = constants.%float.132]
// CHECK:STDOUT:   %Add.ref.loc23: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc23_33: Core.FloatLiteral = float_literal_value 10e19 [concrete = constants.%float.e08d73.1]
// CHECK:STDOUT:   %float.loc23_41: Core.FloatLiteral = float_literal_value 10e-21 [concrete = constants.%float.20e171.1]
// CHECK:STDOUT:   %Add.call.loc23: init Core.FloatLiteral = call %Add.ref.loc23(%float.loc23_33, %float.loc23_41) [concrete = constants.%float.ed50b1.1]
// CHECK:STDOUT:   %Add.ref.loc24: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc24_33: Core.FloatLiteral = float_literal_value 10e-21 [concrete = constants.%float.20e171.2]
// CHECK:STDOUT:   %float.loc24_42: Core.FloatLiteral = float_literal_value 10e19 [concrete = constants.%float.e08d73.2]
// CHECK:STDOUT:   %Add.call.loc24: init Core.FloatLiteral = call %Add.ref.loc24(%float.loc24_33, %float.loc24_42) [concrete = constants.%float.ed50b1.2]
// CHECK:STDOUT:   %Add.ref.loc25: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc25_33: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.4]
// CHECK:STDOUT:   %float.loc25_38: Core.FloatLiteral = float_literal_value 1000000000000000000000000000000e-30 [concrete = constants.%float.54c]
// CHECK:STDOUT:   %Add.call.loc25: init Core.FloatLiteral = call %Add.ref.loc25(%float.loc25_33, %float.loc25_38) [concrete = constants.%float.fc1]
// CHECK:STDOUT:   %Add.ref.loc26: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc26_33: Core.FloatLiteral = float_literal_value 9223372036854775807e-19 [concrete = constants.%float.09a]
// CHECK:STDOUT:   %float.loc26_56: Core.FloatLiteral = float_literal_value 1e-19 [concrete = constants.%float.fbf8ca.1]
// CHECK:STDOUT:   %Add.call.loc26: init Core.FloatLiteral = call %Add.ref.loc26(%float.loc26_33, %float.loc26_56) [concrete = constants.%float.36a]
// CHECK:STDOUT:   %Add.ref.loc27: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc27_33: Core.FloatLiteral = float_literal_value 16p1023999999996 [concrete = constants.%float.b31]
// CHECK:STDOUT:   %float.loc27_54: Core.FloatLiteral = float_literal_value 1e-19 [concrete = constants.%float.fbf8ca.2]
// CHECK:STDOUT:   %Add.call.loc27: init Core.FloatLiteral = call %Add.ref.loc27(%float.loc27_33, %float.loc27_54) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
