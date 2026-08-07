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
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f1: Core.FloatLiteral = Negate(1.0);
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f1: Core.FloatLiteral = Negate(1.0);
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f2: Core.FloatLiteral = Negate(1.8e9);
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f2: Core.FloatLiteral = Negate(1.8e9);

// Negative values become positive
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f3: Core.FloatLiteral = Negate(Negate(87.87));
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f3: Core.FloatLiteral = Negate(Negate(87.87));
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f4: Core.FloatLiteral = Negate(Negate(0x1.8p1));
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f4: Core.FloatLiteral = Negate(Negate(0x1.8p1));

// Zero is still zero
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f5: Core.FloatLiteral = Negate(0.0);
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f5: Core.FloatLiteral = Negate(0.0);
// CHECK:STDERR: negate.carbon:[[@LINE+4]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR: let f6: Core.FloatLiteral = Negate(Negate(0.0));
// CHECK:STDERR:       ^
// CHECK:STDERR:
let f6: Core.FloatLiteral = Negate(Negate(0.0));
//@dump-sem-ir-end

// CHECK:STDOUT: --- negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %f1.patt: %pattern_type = value_binding_pattern f1 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.84a: Core.FloatLiteral = float_literal_value -10e-1 [concrete]
// CHECK:STDOUT:   %f2.patt: %pattern_type = value_binding_pattern f2 [concrete]
// CHECK:STDOUT:   %float.c59: Core.FloatLiteral = float_literal_value 18e8 [concrete]
// CHECK:STDOUT:   %float.838: Core.FloatLiteral = float_literal_value -18e8 [concrete]
// CHECK:STDOUT:   %f3.patt: %pattern_type = value_binding_pattern f3 [concrete]
// CHECK:STDOUT:   %float.b21268.1: Core.FloatLiteral = float_literal_value 8787e-2 [concrete]
// CHECK:STDOUT:   %float.923: Core.FloatLiteral = float_literal_value -8787e-2 [concrete]
// CHECK:STDOUT:   %float.b21268.2: Core.FloatLiteral = float_literal_value 8787e-2 [concrete]
// CHECK:STDOUT:   %f4.patt: %pattern_type = value_binding_pattern f4 [concrete]
// CHECK:STDOUT:   %float.eda197.1: Core.FloatLiteral = float_literal_value 24p-3 [concrete]
// CHECK:STDOUT:   %float.0b3: Core.FloatLiteral = float_literal_value -24p-3 [concrete]
// CHECK:STDOUT:   %float.eda197.2: Core.FloatLiteral = float_literal_value 24p-3 [concrete]
// CHECK:STDOUT:   %f5.patt: %pattern_type = value_binding_pattern f5 [concrete]
// CHECK:STDOUT:   %float.1f75e4.1: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.2: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %f6.patt: %pattern_type = value_binding_pattern f6 [concrete]
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
// CHECK:STDOUT:   %.loc19_39.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc19 [concrete = constants.%float.84a]
// CHECK:STDOUT:   %.loc19_39.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc19, %.loc19_39.1 [concrete = constants.%float.84a]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %FloatLiteral.ref.loc19 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc19: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f1: Core.FloatLiteral = wrapper_binding f1, %.loc19_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f1.patt: %pattern_type = value_binding_pattern f1 [concrete = constants.%f1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_41.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc20 [concrete = constants.%float.838]
// CHECK:STDOUT:   %.loc20_41.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc20, %.loc20_41.1 [concrete = constants.%float.838]
// CHECK:STDOUT:   %.loc20_13: type = splice_block %FloatLiteral.ref.loc20 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc20: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f2: Core.FloatLiteral = wrapper_binding f2, %.loc20_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f2.patt: %pattern_type = value_binding_pattern f2 [concrete = constants.%f2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_49.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc23_49 [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %.loc23_49.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc23_49, %.loc23_49.1 [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %FloatLiteral.ref.loc23 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc23: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f3: Core.FloatLiteral = wrapper_binding f3, %.loc23_49.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f3.patt: %pattern_type = value_binding_pattern f3 [concrete = constants.%f3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_51.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc24_51 [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %.loc24_51.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc24_51, %.loc24_51.1 [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %.loc24_13: type = splice_block %FloatLiteral.ref.loc24 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc24: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f4: Core.FloatLiteral = wrapper_binding f4, %.loc24_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f4.patt: %pattern_type = value_binding_pattern f4 [concrete = constants.%f4.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_39.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc27 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %.loc27_39.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc27, %.loc27_39.1 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %.loc27_13: type = splice_block %FloatLiteral.ref.loc27 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc27: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f5: Core.FloatLiteral = wrapper_binding f5, %.loc27_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f5.patt: %pattern_type = value_binding_pattern f5 [concrete = constants.%f5.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_47.1: Core.FloatLiteral = value_of_initializer @__global_init.%Negate.call.loc28_47 [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   %.loc28_47.2: Core.FloatLiteral = converted @__global_init.%Negate.call.loc28_47, %.loc28_47.1 [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   %.loc28_13: type = splice_block %FloatLiteral.ref.loc28 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc28: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc28: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f6: Core.FloatLiteral = wrapper_binding f6, %.loc28_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f6.patt: %pattern_type = value_binding_pattern f6 [concrete = constants.%f6.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc19: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc19: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %Negate.call.loc19: init Core.FloatLiteral = call %Negate.ref.loc19(%float.loc19) [concrete = constants.%float.84a]
// CHECK:STDOUT:   %Negate.ref.loc20: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc20: Core.FloatLiteral = float_literal_value 18e8 [concrete = constants.%float.c59]
// CHECK:STDOUT:   %Negate.call.loc20: init Core.FloatLiteral = call %Negate.ref.loc20(%float.loc20) [concrete = constants.%float.838]
// CHECK:STDOUT:   %Negate.ref.loc23_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc23_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc23: Core.FloatLiteral = float_literal_value 8787e-2 [concrete = constants.%float.b21268.1]
// CHECK:STDOUT:   %Negate.call.loc23_48: init Core.FloatLiteral = call %Negate.ref.loc23_36(%float.loc23) [concrete = constants.%float.923]
// CHECK:STDOUT:   %.loc23_48.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc23_48 [concrete = constants.%float.923]
// CHECK:STDOUT:   %.loc23_48.2: Core.FloatLiteral = converted %Negate.call.loc23_48, %.loc23_48.1 [concrete = constants.%float.923]
// CHECK:STDOUT:   %Negate.call.loc23_49: init Core.FloatLiteral = call %Negate.ref.loc23_29(%.loc23_48.2) [concrete = constants.%float.b21268.2]
// CHECK:STDOUT:   %Negate.ref.loc24_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc24_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc24: Core.FloatLiteral = float_literal_value 24p-3 [concrete = constants.%float.eda197.1]
// CHECK:STDOUT:   %Negate.call.loc24_50: init Core.FloatLiteral = call %Negate.ref.loc24_36(%float.loc24) [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %.loc24_50.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc24_50 [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %.loc24_50.2: Core.FloatLiteral = converted %Negate.call.loc24_50, %.loc24_50.1 [concrete = constants.%float.0b3]
// CHECK:STDOUT:   %Negate.call.loc24_51: init Core.FloatLiteral = call %Negate.ref.loc24_29(%.loc24_50.2) [concrete = constants.%float.eda197.2]
// CHECK:STDOUT:   %Negate.ref.loc27: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc27: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.1]
// CHECK:STDOUT:   %Negate.call.loc27: init Core.FloatLiteral = call %Negate.ref.loc27(%float.loc27) [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %Negate.ref.loc28_29: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc28_36: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc28: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.3]
// CHECK:STDOUT:   %Negate.call.loc28_46: init Core.FloatLiteral = call %Negate.ref.loc28_36(%float.loc28) [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %.loc28_46.1: Core.FloatLiteral = value_of_initializer %Negate.call.loc28_46 [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %.loc28_46.2: Core.FloatLiteral = converted %Negate.call.loc28_46, %.loc28_46.1 [concrete = constants.%float.1f75e4.4]
// CHECK:STDOUT:   %Negate.call.loc28_47: init Core.FloatLiteral = call %Negate.ref.loc28_29(%.loc28_46.2) [concrete = constants.%float.1f75e4.5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
