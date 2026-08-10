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

// --- float_literal_sub.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.FloatLiteral) -> Core.FloatLiteral = "float.negate";
fn Sub(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.sub";

// Since FloatLiterals are not canonical we need to compare using sized float.
class Expect(N: f128) {}
fn Test(generic N: Core.FloatLiteral) -> Expect(N) { return {}; }

//@dump-sem-ir-begin
let v1: Core.FloatLiteral = Sub(1.0, 2.0);
let v2: Core.FloatLiteral = Sub(0x1000.0p10, 0x1.0p10);
//@dump-sem-ir-end


fn F() {
    Test(Sub(2.0, 1.0)) as Expect(1.0);
    Test(Sub(0x1000.0p10, 0x1.0p10)) as Expect(0xFFF.0p10);
    Test(Sub(0x1.8p0, 2.5)) as Expect(Negate(1.0));
    Test(Sub(2.5, 0x1.8p0)) as Expect(1.0);
    Test(Sub(1.0e20, 1.0e-20)) as Expect(0.9999999999999999999999999999999999999999e20);
    Test(Sub(1.0e-20, 1.0e20)) as Expect(Negate(0.9999999999999999999999999999999999999999e20));
    Test(Sub(1.0, 1.000000000000000000000000000001)) as Expect(Negate(0.000000000000000000000000000001));
    Test(Sub(1.0000000000000000000000000000001, 1.0)) as Expect(0.0000000000000000000000000000001);
}

// --- fail_too_large.carbon

library "[[@TEST_NAME]]";

fn Sub(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.sub";

// CHECK:STDERR: fail_too_large.carbon:[[@LINE+2]]:29-75: error: binary calculation `16*2^1023999999996 - 1*10^-19` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
// CHECK:STDERR:
let v1: Core.FloatLiteral = Sub(0x1.0p1024000000000, 0.0000000000000000001);

// CHECK:STDOUT: --- float_literal_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.dab: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [concrete]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %v1.patt: %pattern_type.dab = value_binding_pattern v1 [concrete]
// CHECK:STDOUT:   %float.6daae3.1: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.cceb42.1: Core.FloatLiteral = float_literal_value 20e-1 [concrete]
// CHECK:STDOUT:   %float.e85: Core.FloatLiteral = float_literal_value -10e-1 [concrete]
// CHECK:STDOUT:   %v2.patt: %pattern_type.dab = value_binding_pattern v2 [concrete]
// CHECK:STDOUT:   %float.8466d5.1: Core.FloatLiteral = float_literal_value 65536p6 [concrete]
// CHECK:STDOUT:   %float.333a1f.1: Core.FloatLiteral = float_literal_value 16p6 [concrete]
// CHECK:STDOUT:   %float.bec4a4.1: Core.FloatLiteral = float_literal_value 65520p6 [concrete]
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
// CHECK:STDOUT:   %.loc12_41.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc12 [concrete = constants.%float.e85]
// CHECK:STDOUT:   %.loc12_41.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc12, %.loc12_41.1 [concrete = constants.%float.e85]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %FloatLiteral.ref.loc12 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc12: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v1: Core.FloatLiteral = wrapper_binding v1, %.loc12_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v1.patt: %pattern_type.dab = value_binding_pattern v1 [concrete = constants.%v1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_54.1: Core.FloatLiteral = value_of_initializer @__global_init.%Sub.call.loc13 [concrete = constants.%float.bec4a4.1]
// CHECK:STDOUT:   %.loc13_54.2: Core.FloatLiteral = converted @__global_init.%Sub.call.loc13, %.loc13_54.1 [concrete = constants.%float.bec4a4.1]
// CHECK:STDOUT:   %.loc13_13: type = splice_block %FloatLiteral.ref.loc13 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %FloatLiteral.ref.loc13: type = name_ref FloatLiteral, imports.%Core.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: Core.FloatLiteral = wrapper_binding v2, %.loc13_54.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %pattern_type.dab = value_binding_pattern v2 [concrete = constants.%v2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc12: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc12_33: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6daae3.1]
// CHECK:STDOUT:   %float.loc12_38: Core.FloatLiteral = float_literal_value 20e-1 [concrete = constants.%float.cceb42.1]
// CHECK:STDOUT:   %Sub.call.loc12: init Core.FloatLiteral = call %Sub.ref.loc12(%float.loc12_33, %float.loc12_38) [concrete = constants.%float.e85]
// CHECK:STDOUT:   %Sub.ref.loc13: %Sub.type = name_ref Sub, file.%Sub.decl [concrete = constants.%Sub]
// CHECK:STDOUT:   %float.loc13_33: Core.FloatLiteral = float_literal_value 65536p6 [concrete = constants.%float.8466d5.1]
// CHECK:STDOUT:   %float.loc13_46: Core.FloatLiteral = float_literal_value 16p6 [concrete = constants.%float.333a1f.1]
// CHECK:STDOUT:   %Sub.call.loc13: init Core.FloatLiteral = call %Sub.ref.loc13(%float.loc13_33, %float.loc13_46) [concrete = constants.%float.bec4a4.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
