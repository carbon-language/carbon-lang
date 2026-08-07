// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/negate.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.FloatLiteral) -> Core.FloatLiteral = "float.negate";

//@dump-sem-ir-begin
// Positive values become negative
let v1: Core.FloatLiteral = Negate(1.0);
let v2: Core.FloatLiteral = Negate(1.8e9);

// Negative values become positive
let v3: Core.FloatLiteral = Negate(Negate(87.87));
let v4: Core.FloatLiteral = Negate(Negate(0x1.8p1));

// Zero is still zero
let v5: Core.FloatLiteral = Negate(0.0);
let v6: Core.FloatLiteral = Negate(Negate(0.0));
//@dump-sem-ir-end

// CHECK:STDOUT: --- negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %v1.patt: %pattern_type = value_binding_pattern v1 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.84a: Core.FloatLiteral = float_literal_value -10e-1 [concrete]
// CHECK:STDOUT:   %v2.patt: %pattern_type = value_binding_pattern v2 [concrete]
// CHECK:STDOUT:   %float.c59: Core.FloatLiteral = float_literal_value 18e8 [concrete]
// CHECK:STDOUT:   %float.838: Core.FloatLiteral = float_literal_value -18e8 [concrete]
// CHECK:STDOUT:   %v3.patt: %pattern_type = value_binding_pattern v3 [concrete]
// CHECK:STDOUT:   %float.b21268.1: Core.FloatLiteral = float_literal_value 8787e-2 [concrete]
// CHECK:STDOUT:   %float.923: Core.FloatLiteral = float_literal_value -8787e-2 [concrete]
// CHECK:STDOUT:   %float.b21268.2: Core.FloatLiteral = float_literal_value 8787e-2 [concrete]
// CHECK:STDOUT:   %v4.patt: %pattern_type = value_binding_pattern v4 [concrete]
// CHECK:STDOUT:   %float.eda197.1: Core.FloatLiteral = float_literal_value 24p-3 [concrete]
// CHECK:STDOUT:   %float.0b3: Core.FloatLiteral = float_literal_value -24p-3 [concrete]
// CHECK:STDOUT:   %float.eda197.2: Core.FloatLiteral = float_literal_value 24p-3 [concrete]
// CHECK:STDOUT:   %v5.patt: %pattern_type = value_binding_pattern v5 [concrete]
// CHECK:STDOUT:   %float.1f75e4.1: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.2: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %v6.patt: %pattern_type = value_binding_pattern v6 [concrete]
// CHECK:STDOUT:   %float.1f75e4.3: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.4: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.5: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
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
// CHECK:STDOUT:   %.loc23_39.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc23 [concrete = constants.%float.84a]
// CHECK:STDOUT:   %.loc23_39.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc23, %.loc23_39.1 [concrete = constants.%float.84a]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %FloatLiteral.ref.loc23 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc23: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v1: Core.FloatLiteral = wrapper_binding v1, %.loc23_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v1.patt: %pattern_type = value_binding_pattern v1 [concrete = constants.%v1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_41.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc28 [concrete = constants.%float.838]
// CHECK:STDOUT:   %.loc28_41.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc28, %.loc28_41.1 [concrete = constants.%float.838]
// CHECK:STDOUT:   %.loc28_13: type = splice_block %FloatLiteral.ref.loc28 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc28: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc28: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: Core.FloatLiteral = wrapper_binding v2, %.loc28_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %pattern_type = value_binding_pattern v2 [concrete = constants.%v2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35_49.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc35_49 [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %.loc35_49.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc35_49, %.loc35_49.1 [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %.loc35_13: type = splice_block %FloatLiteral.ref.loc35 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc35: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc35: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3: Core.FloatLiteral = wrapper_binding v3, %.loc35_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v3.patt: %pattern_type = value_binding_pattern v3 [concrete = constants.%v3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc40_51.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc40_51 [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %.loc40_51.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc40_51, %.loc40_51.1 [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %.loc40_13: type = splice_block %FloatLiteral.ref.loc40 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc40: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc40: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4: Core.FloatLiteral = wrapper_binding v4, %.loc40_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v4.patt: %pattern_type = value_binding_pattern v4 [concrete = constants.%v4.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47_39.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc47 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %.loc47_39.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc47, %.loc47_39.1 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %.loc47_13: type = splice_block %FloatLiteral.ref.loc47 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc47: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc47: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5: Core.FloatLiteral = wrapper_binding v5, %.loc47_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v5.patt: %pattern_type = value_binding_pattern v5 [concrete = constants.%v5.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc52_47.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc52_47 [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   %.loc52_47.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc52_47, %.loc52_47.1 [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   %.loc52_13: type = splice_block %FloatLiteral.ref.loc52 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc52: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc52: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v6: Core.FloatLiteral = wrapper_binding v6, %.loc52_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v6.patt: %pattern_type = value_binding_pattern v6 [concrete = constants.%v6.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc23: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc23: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %Negate.call.loc23: init Core.FloatLiteral = call %Negate.ref.loc23(%float.loc23) [concrete = constants.%float.84a]
// CHECK:STDOUT:   %Negate.ref.loc28: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc28: Core.FloatLiteral = float_literal_value 18e8 [concrete = constants.%float.c59]
// CHECK:STDOUT:   %Negate.call.loc28: init Core.FloatLiteral = call %Negate.ref.loc28(%float.loc28) [concrete = constants.%float.838]
// CHECK:STDOUT:   %Negate.ref.loc35_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc35_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc35: Core.FloatLiteral = float_literal_value 8787e-2 [concrete = constants.%float.b21268.1]
// CHECK:STDOUT:   %Negate.call.loc35_48: init Core.FloatLiteral = call %Negate.ref.loc35_36(%float.loc35) [concrete = constants.%float.923]
// CHECK:STDOUT:   %.loc35_48.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc35_48 [concrete = constants.%float.923]
// CHECK:STDOUT:   %.loc35_48.2: Core.FloatLiteral = converted %Negate.call.loc35_48, %.loc35_48.1 [concrete = constants.%float.923]
// CHECK:STDOUT:   %Negate.call.loc35_49: init Core.FloatLiteral = call %Negate.ref.loc35_29(%.loc35_48.2) [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %Negate.ref.loc40_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc40_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc40: Core.FloatLiteral = float_literal_value 24p-3 [concrete = constants.%float.eda197.1]
// CHECK:STDOUT:   %Negate.call.loc40_50: init Core.FloatLiteral = call %Negate.ref.loc40_36(%float.loc40) [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %.loc40_50.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc40_50 [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %.loc40_50.2: Core.FloatLiteral = converted %Negate.call.loc40_50, %.loc40_50.1 [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %Negate.call.loc40_51: init Core.FloatLiteral = call %Negate.ref.loc40_29(%.loc40_50.2) [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %Negate.ref.loc47: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc47: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.1]
// CHECK:STDOUT:   %Negate.call.loc47: init Core.FloatLiteral = call %Negate.ref.loc47(%float.loc47) [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %Negate.ref.loc52_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc52_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc52: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.3]
// CHECK:STDOUT:   %Negate.call.loc52_46: init Core.FloatLiteral = call %Negate.ref.loc52_36(%float.loc52) [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %.loc52_46.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc52_46 [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %.loc52_46.2: Core.FloatLiteral = converted %Negate.call.loc52_46, %.loc52_46.1 [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %Negate.call.loc52_47: init Core.FloatLiteral = call %Negate.ref.loc52_29(%.loc52_46.2) [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
