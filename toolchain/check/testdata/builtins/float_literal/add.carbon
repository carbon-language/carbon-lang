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

// --- float_literal_add.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.add";

// Since FloatLiterals are not canonical we need to compare using sized float.
class Expect(N: f128) {}
fn Test(generic N: Core.FloatLiteral) -> Expect(N) { return {}; }

//@dump-sem-ir-begin
let v1: Core.FloatLiteral = Add(0x1.0000000000, 100.0);
let v2: Core.FloatLiteral = Add(0x1000.0p10, 0x1.0p10);
//@dump-sem-ir-end

fn F() {
    Test(Add(1.0, 1.0)) as Expect(2.0);
    Test(Add(0x1.0p0, 1.0)) as Expect(2.0);
    Test(Add(0x1.8p0, 2.5)) as Expect(4.0);
    Test(Add(1.0e20, 1.0e-20)) as Expect(1.00000000000000000000000000000000000000001e20);
    Test(Add(1.0e-20, 1.0e20)) as Expect(1.00000000000000000000000000000000000000001e20);
    Test(Add(1.0, 1.000000000000000000000000000000)) as Expect(2.0);
    Test(Add(0.922337203685477580, 0.0000000000000000001)) as Expect(0.9223372036854775801);
}

// --- fail_too_large.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.add";

fn F(){
     // CHECK:STDERR: fail_too_large.carbon:[[@LINE+2]]:6-52: error: binary calculation `16*2^1023999999996 + 1*10^-19` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
     // CHECK:STDERR:
     Add(0x1.0p1024000000000, 0.0000000000000000001);
}

// CHECK:STDOUT: --- float_literal_add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.dab: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [concrete]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %v1.patt: %pattern_type.dab = value_binding_pattern v1 [concrete]
// CHECK:STDOUT:   %float.b0f: Core.FloatLiteral = float_literal_value 1099511627776p-40 [concrete]
// CHECK:STDOUT:   %float.471: Core.FloatLiteral = float_literal_value 1000e-1 [concrete]
// CHECK:STDOUT:   %float.9f6: Core.FloatLiteral = float_literal_value 1010000000000000000000000000000000000000000e-40 [concrete]
// CHECK:STDOUT:   %v2.patt: %pattern_type.dab = value_binding_pattern v2 [concrete]
// CHECK:STDOUT:   %float.846: Core.FloatLiteral = float_literal_value 65536p6 [concrete]
// CHECK:STDOUT:   %float.333: Core.FloatLiteral = float_literal_value 16p6 [concrete]
// CHECK:STDOUT:   %float.283: Core.FloatLiteral = float_literal_value 65552p6 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .FloatLiteral = %Core.FloatLiteral
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.FloatLiteral: type = import_ref Core//prelude/parts/float_literal, FloatLiteral, loaded [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc11_54.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc11 [concrete = constants.%float.9f6]
// CHECK:STDOUT:   %.loc11_54.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc11, %.loc11_54.1 [concrete = constants.%float.9f6]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %FloatLiteral.ref.loc11 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc11: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v1: Core.FloatLiteral = wrapper_binding v1, %.loc11_54.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v1.patt: %pattern_type.dab = value_binding_pattern v1 [concrete = constants.%v1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_54.1: Core.FloatLiteral = value_of_initializer @__global_init.%Add.call.loc12 [concrete = constants.%float.283]
// CHECK:STDOUT:   %.loc12_54.2: Core.FloatLiteral = converted @__global_init.%Add.call.loc12, %.loc12_54.1 [concrete = constants.%float.283]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %FloatLiteral.ref.loc12 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc12: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: Core.FloatLiteral = wrapper_binding v2, %.loc12_54.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %pattern_type.dab = value_binding_pattern v2 [concrete = constants.%v2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref.loc11: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc11_33: Core.FloatLiteral = float_literal_value 1099511627776p-40 [concrete = constants.%float.b0f]
// CHECK:STDOUT:   %float.loc11_49: Core.FloatLiteral = float_literal_value 1000e-1 [concrete = constants.%float.471]
// CHECK:STDOUT:   %Add.call.loc11: init Core.FloatLiteral = call %Add.ref.loc11(%float.loc11_33, %float.loc11_49) [concrete = constants.%float.9f6]
// CHECK:STDOUT:   %Add.ref.loc12: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %float.loc12_33: Core.FloatLiteral = float_literal_value 65536p6 [concrete = constants.%float.846]
// CHECK:STDOUT:   %float.loc12_46: Core.FloatLiteral = float_literal_value 16p6 [concrete = constants.%float.333]
// CHECK:STDOUT:   %Add.call.loc12: init Core.FloatLiteral = call %Add.ref.loc12(%float.loc12_33, %float.loc12_46) [concrete = constants.%float.283]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
