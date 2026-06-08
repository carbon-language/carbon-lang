// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert_float.carbon

// --- int_to_float.carbon
library "[[@TEST_NAME]]";

fn ConvertI32ToF64(a: i32) -> f64 = "int.convert_float";
fn ConvertU32ToF64(a: u32) -> f64 = "int.convert_float";
fn ConvertIntLiteralToF64(a: Core.IntLiteral) -> f64 = "int.convert_float";
fn ConvertIntLiteralToFloatLiteral(a: Core.IntLiteral) -> Core.FloatLiteral = "int.convert_float";

fn Negate(a: f64) -> f64 = "float.negate";

class Expect[T:! type](N:! T) {}
fn Test[T:! type](N:! T) -> Expect(N) { return {}; }

// --- identity.carbon
library "[[@TEST_NAME]]";
import library "int_to_float";

fn F() {
  Test(ConvertI32ToF64(0)) as Expect(0.0 as f64);
  Test(ConvertI32ToF64(1)) as Expect(1.0 as f64);
  Test(ConvertI32ToF64(-1)) as Expect(Negate(1.0));
  Test(ConvertI32ToF64(123456789)) as Expect(123456789.0 as f64);

  Test(ConvertU32ToF64(0)) as Expect(0.0 as f64);
  Test(ConvertU32ToF64(1)) as Expect(1.0 as f64);
  Test(ConvertU32ToF64(4294967295)) as Expect(4294967295.0 as f64);

  Test(ConvertIntLiteralToF64(1)) as Expect(1.0 as f64);
  Test(ConvertIntLiteralToF64(-1)) as Expect(Negate(1.0));

  Test(ConvertIntLiteralToFloatLiteral(1) as f64) as Expect(1.0 as f64);
}

// --- fail_overflow.carbon
library "[[@TEST_NAME]]";
import library "int_to_float";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:21: error: integer value 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 too large for floating-point type `f64` [IntTooLargeForFloatType]
// CHECK:STDERR: let overflow: f64 = ConvertIntLiteralToF64(1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow: f64 = ConvertIntLiteralToF64(1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000);
//@dump-sem-ir-end

// --- fail_todo_negative_literal.carbon
library "[[@TEST_NAME]]";
import library "int_to_float";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_negative_literal.carbon:[[@LINE+4]]:20: error: semantics TODO: `negative float literal conversion` [SemanticsTodo]
// CHECK:STDERR: let invalid: f64 = ConvertIntLiteralToFloatLiteral(-1) as f64;
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let invalid: f64 = ConvertIntLiteralToFloatLiteral(-1) as f64;
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %overflow.patt: %pattern_type.fb7 = value_binding_pattern overflow [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToF64.type: type = fn_type @ConvertIntLiteralToF64 [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToF64: %ConvertIntLiteralToF64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000: Core.IntLiteral = int_value 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertIntLiteralToF64: %ConvertIntLiteralToF64.type = import_ref Main//int_to_float, ConvertIntLiteralToF64, loaded [concrete = constants.%ConvertIntLiteralToF64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %overflow.patt: %pattern_type.fb7 = value_binding_pattern overflow [concrete = constants.%overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_354.1: %f64.dc1 = value_of_initializer @__global_init.%ConvertIntLiteralToF64.call [concrete = <error>]
// CHECK:STDOUT:   %.loc9_354.2: %f64.dc1 = converted @__global_init.%ConvertIntLiteralToF64.call, %.loc9_354.1 [concrete = <error>]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %overflow: %f64.dc1 = value_binding overflow, %.loc9_354.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertIntLiteralToF64.ref: %ConvertIntLiteralToF64.type = name_ref ConvertIntLiteralToF64, imports.%Main.ConvertIntLiteralToF64 [concrete = constants.%ConvertIntLiteralToF64]
// CHECK:STDOUT:   %int_1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000: Core.IntLiteral = int_value 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 [concrete = constants.%int_1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]
// CHECK:STDOUT:   %ConvertIntLiteralToF64.call: init %f64.dc1 = call %ConvertIntLiteralToF64.ref(%int_1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_negative_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %invalid.patt: %pattern_type.fb7 = value_binding_pattern invalid [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToFloatLiteral.type: type = fn_type @ConvertIntLiteralToFloatLiteral [concrete]
// CHECK:STDOUT:   %ConvertIntLiteralToFloatLiteral: %ConvertIntLiteralToFloatLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.802: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.765: type = fn_type_with_self_type %Negate.WithSelf.Op.type.802, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %As.type.0e0: type = facet_type <@As, @As(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.69a: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.829: %Core.FloatLiteral.as.As.impl.Convert.type.69a = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.115: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.0cf: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.20d: %Core.FloatLiteral.as.As.impl.Convert.type.0cf = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.0e0 = facet_value Core.FloatLiteral, (%As.impl_witness.115) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.4d9: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.dc1, %As.facet) [concrete]
// CHECK:STDOUT:   %.e52: type = fn_type_with_self_type %As.WithSelf.Convert.type.4d9, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.20d, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertIntLiteralToFloatLiteral: %ConvertIntLiteralToFloatLiteral.type = import_ref Main//int_to_float, ConvertIntLiteralToFloatLiteral, loaded [concrete = constants.%ConvertIntLiteralToFloatLiteral]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c06: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.69a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.829)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.c06), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %invalid.patt: %pattern_type.fb7 = value_binding_pattern invalid [concrete = constants.%invalid.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %invalid: %f64.dc1 = value_binding invalid, @__global_init.%.loc9_56.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertIntLiteralToFloatLiteral.ref: %ConvertIntLiteralToFloatLiteral.type = name_ref ConvertIntLiteralToFloatLiteral, imports.%Main.ConvertIntLiteralToFloatLiteral [concrete = constants.%ConvertIntLiteralToFloatLiteral]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.765 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_52: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc9_52(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc9_52.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc9_52.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc9_52.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %ConvertIntLiteralToFloatLiteral.call: init Core.FloatLiteral = call %ConvertIntLiteralToFloatLiteral.ref(%.loc9_52.2) [concrete = <error>]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %impl.elem0: %.e52 = impl_witness_access constants.%As.impl_witness.115, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.20d]
// CHECK:STDOUT:   %bound_method.loc9_56.1: <bound method> = bound_method %ConvertIntLiteralToFloatLiteral.call, %impl.elem0 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_56.2: <bound method> = bound_method %ConvertIntLiteralToFloatLiteral.call, %specific_fn [concrete = <error>]
// CHECK:STDOUT:   %.loc9_54.1: Core.FloatLiteral = value_of_initializer %ConvertIntLiteralToFloatLiteral.call [concrete = <error>]
// CHECK:STDOUT:   %.loc9_54.2: Core.FloatLiteral = converted %ConvertIntLiteralToFloatLiteral.call, %.loc9_54.1 [concrete = <error>]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.dc1 = call %bound_method.loc9_56.2(%.loc9_54.2) [concrete = <error>]
// CHECK:STDOUT:   %.loc9_56.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = <error>]
// CHECK:STDOUT:   %.loc9_56.2: %f64.dc1 = converted %ConvertIntLiteralToFloatLiteral.call, %.loc9_56.1 [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
