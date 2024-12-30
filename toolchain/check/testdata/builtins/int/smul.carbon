// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smul.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn F() {
  Test(Mul(0, 0)) as Expect(0);
  Test(Mul(0, 3)) as Expect(0);
  Test(Mul(1, 2)) as Expect(2);
  Test(Mul(3, 2)) as Expect(6);
  Test(Mul(0x7FFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(Mul(0x7FFF_FFFF, -1)) as Expect(-0x7FFF_FFFF);
}

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mul(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Mul(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.smul";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Mul(0, 0)) as Expect(0);
  Test(Mul(0, 3)) as Expect(0);
  Test(Mul(1, 2)) as Expect(2);
  Test(Mul(3, 2)) as Expect(6);
  Test(Mul(0x7FFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(Mul(0x7FFF_FFFF, -1)) as Expect(-0x7FFF_FFFF);

  // Test some cases that might -- but shouldn't -- overflow.
  Test(Mul(0x8000_0000_0000_0000, 1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(0x8000_0000_0000_0000, -1)) as Expect(-0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, 1)) as Expect(-0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -0x8000_0000_0000_0000))
    as Expect(0x4000_0000_0000_0000_0000_0000_0000_0000);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

let a: i32 = Mul(0x7FFF, 0x10000);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in calculation `32768 * 65536` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Mul(0x8000, 0x10000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
let b: i32 = Mul(0x8000, 0x10000);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mul.type.1: type = fn_type @Mul.1 [template]
// CHECK:STDOUT:   %Mul: %Mul.type.1 = struct_value () [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_2.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_2.2) [template]
// CHECK:STDOUT:   %int_6.1: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_6.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_6.1) [template]
// CHECK:STDOUT:   %int_6.2: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_6.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_2147483647.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_2147483647.2) [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %int_-2147483647.1: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_-2147483647.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_-2147483647.1) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2147483647.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-2147483647.2: Core.IntLiteral = int_value -2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_-2147483647.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Negate = %import_ref.229
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type.1 = fn_decl @Mul.1 [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_11: type = splice_block %i32.loc4_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_19: type = splice_block %i32.loc4_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_14.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc7_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc7_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_19: type = splice_block %i32.loc18_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc18_27: type = splice_block %i32.loc18_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: %i32) {
// CHECK:STDOUT:   %N.loc6_14.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smul";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: %i32) {
// CHECK:STDOUT:   %N.loc7_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc7_29.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) -> %return.param_patt: @Test.%Expect.loc7_29.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_41.2: init @Test.%Expect.loc7_29.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_42: init @Test.%Expect.loc7_29.2 (%Expect.1) = converted %.loc7_41.1, %.loc7_41.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_42 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc10: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_0.loc10_12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc10_15: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_12: <bound method> = bound_method %int_0.loc10_12, %impl.elem0.loc10_12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_12: <specific function> = specific_function %Convert.bound.loc10_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_12: init %i32 = call %Convert.specific_fn.loc10_12(%int_0.loc10_12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_12.1: %i32 = value_of_initializer %int.convert_checked.loc10_12 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_12.2: %i32 = converted %int_0.loc10_12, %.loc10_12.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_15: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_15: <bound method> = bound_method %int_0.loc10_15, %impl.elem0.loc10_15 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_15: <specific function> = specific_function %Convert.bound.loc10_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_15: init %i32 = call %Convert.specific_fn.loc10_15(%int_0.loc10_15) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_15.1: %i32 = value_of_initializer %int.convert_checked.loc10_15 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = converted %int_0.loc10_15, %.loc10_15.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.smul.loc10: init %i32 = call %Mul.ref.loc10(%.loc10_12.2, %.loc10_15.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_17.1: %i32 = value_of_initializer %int.smul.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_17.2: %i32 = converted %int.smul.loc10, %.loc10_17.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_29: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_30: <bound method> = bound_method %int_0.loc10_29, %impl.elem0.loc10_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_30: <specific function> = specific_function %Convert.bound.loc10_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_30: init %i32 = call %Convert.specific_fn.loc10_30(%int_0.loc10_29) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %int.convert_checked.loc10_30 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_0.loc10_29, %.loc10_30.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc11: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_0.loc11_12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_3.loc11: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc11_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_12: <bound method> = bound_method %int_0.loc11_12, %impl.elem0.loc11_12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_12: <specific function> = specific_function %Convert.bound.loc11_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_12: init %i32 = call %Convert.specific_fn.loc11_12(%int_0.loc11_12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_12.1: %i32 = value_of_initializer %int.convert_checked.loc11_12 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_12.2: %i32 = converted %int_0.loc11_12, %.loc11_12.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc11_15: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_15: <bound method> = bound_method %int_3.loc11, %impl.elem0.loc11_15 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_15: <specific function> = specific_function %Convert.bound.loc11_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_15: init %i32 = call %Convert.specific_fn.loc11_15(%int_3.loc11) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc11_15.1: %i32 = value_of_initializer %int.convert_checked.loc11_15 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc11_15.2: %i32 = converted %int_3.loc11, %.loc11_15.1 [template = constants.%int_3.2]
// CHECK:STDOUT:   %int.smul.loc11: init %i32 = call %Mul.ref.loc11(%.loc11_12.2, %.loc11_15.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_17.1: %i32 = value_of_initializer %int.smul.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_17.2: %i32 = converted %int.smul.loc11, %.loc11_17.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_17.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11_29: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_30: <bound method> = bound_method %int_0.loc11_29, %impl.elem0.loc11_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_30: <specific function> = specific_function %Convert.bound.loc11_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_30: init %i32 = call %Convert.specific_fn.loc11_30(%int_0.loc11_29) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_30.1: %i32 = value_of_initializer %int.convert_checked.loc11_30 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_30.2: %i32 = converted %int_0.loc11_29, %.loc11_30.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_17.4: ref %Expect.2 = temporary %.loc11_17.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc12: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc12_15: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc12_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_12: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_12 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_12: <specific function> = specific_function %Convert.bound.loc12_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_12: init %i32 = call %Convert.specific_fn.loc12_12(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_12.1: %i32 = value_of_initializer %int.convert_checked.loc12_12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_12.2: %i32 = converted %int_1.loc12, %.loc12_12.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_15: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_15: <bound method> = bound_method %int_2.loc12_15, %impl.elem0.loc12_15 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_15: <specific function> = specific_function %Convert.bound.loc12_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc12_15: init %i32 = call %Convert.specific_fn.loc12_15(%int_2.loc12_15) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_15.1: %i32 = value_of_initializer %int.convert_checked.loc12_15 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_15.2: %i32 = converted %int_2.loc12_15, %.loc12_15.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.smul.loc12: init %i32 = call %Mul.ref.loc12(%.loc12_12.2, %.loc12_15.2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_17.1: %i32 = value_of_initializer %int.smul.loc12 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_17.2: %i32 = converted %int.smul.loc12, %.loc12_17.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_2.2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_17.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc12_29: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc12_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_30: <bound method> = bound_method %int_2.loc12_29, %impl.elem0.loc12_30 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_30: <specific function> = specific_function %Convert.bound.loc12_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc12_30: init %i32 = call %Convert.specific_fn.loc12_30(%int_2.loc12_29) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_30.1: %i32 = value_of_initializer %int.convert_checked.loc12_30 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_30.2: %i32 = converted %int_2.loc12_29, %.loc12_30.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_2.2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_17.4: ref %Expect.3 = temporary %.loc12_17.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc13: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_3.loc13: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc13_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_12: <bound method> = bound_method %int_3.loc13, %impl.elem0.loc13_12 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_12: <specific function> = specific_function %Convert.bound.loc13_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_12: init %i32 = call %Convert.specific_fn.loc13_12(%int_3.loc13) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_12.1: %i32 = value_of_initializer %int.convert_checked.loc13_12 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_12.2: %i32 = converted %int_3.loc13, %.loc13_12.1 [template = constants.%int_3.2]
// CHECK:STDOUT:   %impl.elem0.loc13_15: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_15: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_15 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_15: <specific function> = specific_function %Convert.bound.loc13_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc13_15: init %i32 = call %Convert.specific_fn.loc13_15(%int_2.loc13) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_15.1: %i32 = value_of_initializer %int.convert_checked.loc13_15 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_15.2: %i32 = converted %int_2.loc13, %.loc13_15.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.smul.loc13: init %i32 = call %Mul.ref.loc13(%.loc13_12.2, %.loc13_15.2) [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc13_17.1: %i32 = value_of_initializer %int.smul.loc13 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc13_17.2: %i32 = converted %int.smul.loc13, %.loc13_17.1 [template = constants.%int_6.1]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_6.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc13_17.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.4 = call %Test.specific_fn.loc13() to %.loc13_17.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.2]
// CHECK:STDOUT:   %impl.elem0.loc13_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_30: <bound method> = bound_method %int_6, %impl.elem0.loc13_30 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_30: <specific function> = specific_function %Convert.bound.loc13_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc13_30: init %i32 = call %Convert.specific_fn.loc13_30(%int_6) [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc13_30.1: %i32 = value_of_initializer %int.convert_checked.loc13_30 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc13_30.2: %i32 = converted %int_6, %.loc13_30.1 [template = constants.%int_6.1]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_6.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc13_17.4: ref %Expect.4 = temporary %.loc13_17.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc14: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_2147483647.loc14_12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_12: <bound method> = bound_method %int_2147483647.loc14_12, %impl.elem0.loc14_12 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_12: <specific function> = specific_function %Convert.bound.loc14_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc14_12: init %i32 = call %Convert.specific_fn.loc14_12(%int_2147483647.loc14_12) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_12.1: %i32 = value_of_initializer %int.convert_checked.loc14_12 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_12.2: %i32 = converted %int_2147483647.loc14_12, %.loc14_12.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc14_25: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_25: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_25 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_25: <specific function> = specific_function %Convert.bound.loc14_25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_25: init %i32 = call %Convert.specific_fn.loc14_25(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_25.1: %i32 = value_of_initializer %int.convert_checked.loc14_25 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_25.2: %i32 = converted %int_1.loc14, %.loc14_25.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.smul.loc14: init %i32 = call %Mul.ref.loc14(%.loc14_12.2, %.loc14_25.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_27.1: %i32 = value_of_initializer %int.smul.loc14 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_27.2: %i32 = converted %int.smul.loc14, %.loc14_27.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_2147483647.2) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc14_27.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.5 = call %Test.specific_fn.loc14() to %.loc14_27.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc14_39: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc14_50: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_50: <bound method> = bound_method %int_2147483647.loc14_39, %impl.elem0.loc14_50 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_50: <specific function> = specific_function %Convert.bound.loc14_50, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc14_50: init %i32 = call %Convert.specific_fn.loc14_50(%int_2147483647.loc14_39) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_50.1: %i32 = value_of_initializer %int.convert_checked.loc14_50 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_50.2: %i32 = converted %int_2147483647.loc14_39, %.loc14_50.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_2147483647.2) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc14_27.4: ref %Expect.5 = temporary %.loc14_27.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc15: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_2147483647.loc15_12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_25.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15_25: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_25.1 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_25: init Core.IntLiteral = call %Op.bound.loc15_25(%int_1.loc15) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_12: <bound method> = bound_method %int_2147483647.loc15_12, %impl.elem0.loc15_12 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_12: <specific function> = specific_function %Convert.bound.loc15_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc15_12: init %i32 = call %Convert.specific_fn.loc15_12(%int_2147483647.loc15_12) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc15_12.1: %i32 = value_of_initializer %int.convert_checked.loc15_12 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc15_12.2: %i32 = converted %int_2147483647.loc15_12, %.loc15_12.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc15_25.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_25: <bound method> = bound_method %int.snegate.loc15_25, %impl.elem0.loc15_25.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_25: <specific function> = specific_function %Convert.bound.loc15_25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc15_25.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_25 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_25.2: Core.IntLiteral = converted %int.snegate.loc15_25, %.loc15_25.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_25: init %i32 = call %Convert.specific_fn.loc15_25(%.loc15_25.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_25.3: %i32 = value_of_initializer %int.convert_checked.loc15_25 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_25.4: %i32 = converted %int.snegate.loc15_25, %.loc15_25.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.smul.loc15: init %i32 = call %Mul.ref.loc15(%.loc15_12.2, %.loc15_25.4) [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %.loc15_28.1: %i32 = value_of_initializer %int.smul.loc15 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %.loc15_28.2: %i32 = converted %int.smul.loc15, %.loc15_28.1 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-2147483647.1) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc15_28.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.6 = call %Test.specific_fn.loc15() to %.loc15_28.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc15_41: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc15_40: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15_40: <bound method> = bound_method %int_2147483647.loc15_41, %impl.elem0.loc15_40 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc15_40: init Core.IntLiteral = call %Op.bound.loc15_40(%int_2147483647.loc15_41) [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc15_52: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_52: <bound method> = bound_method %int.snegate.loc15_40, %impl.elem0.loc15_52 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_52: <specific function> = specific_function %Convert.bound.loc15_52, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %.loc15_40.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_40 [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %.loc15_40.2: Core.IntLiteral = converted %int.snegate.loc15_40, %.loc15_40.1 [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_52: init %i32 = call %Convert.specific_fn.loc15_52(%.loc15_40.2) [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %.loc15_52.1: %i32 = value_of_initializer %int.convert_checked.loc15_52 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %.loc15_52.2: %i32 = converted %int.snegate.loc15_40, %.loc15_52.1 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-2147483647.1) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc15_28.4: ref %Expect.6 = temporary %.loc15_28.3, %Test.call.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smul: init %i32 = call %Mul.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc19_19.1: %i32 = value_of_initializer %int.smul
// CHECK:STDOUT:   %.loc19_19.2: %i32 = converted %int.smul, %.loc19_19.1
// CHECK:STDOUT:   return %.loc19_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%N.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_6.1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_6.1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_6.1
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_6.1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_6.1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_6.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2147483647.1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-2147483647.1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-2147483647.1
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2147483647.1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-2147483647.1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-2147483647.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Mul.type.1: type = fn_type @Mul.1 [template]
// CHECK:STDOUT:   %Mul: %Mul.type.1 = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_2) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_6) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_6) [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_2147483647) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_2147483647) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %int_-2147483647: Core.IntLiteral = int_value -2147483647 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_-2147483647) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_-2147483647) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2147483647, %Op.2 [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_-9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.7: <specific function> = specific_function %Test, @Test(%int_-9223372036854775808) [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_9223372036854775808, %Op.2 [template]
// CHECK:STDOUT:   %int_85070591730234615865843651857942052864: Core.IntLiteral = int_value 85070591730234615865843651857942052864 [template]
// CHECK:STDOUT:   %Expect.9: type = class_type @Expect, @Expect(%int_85070591730234615865843651857942052864) [template]
// CHECK:STDOUT:   %Test.specific_fn.8: <specific function> = specific_function %Test, @Test(%int_85070591730234615865843651857942052864) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.8: %Expect.8 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.9: %Expect.9 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type.1 = fn_decl @Mul.1 [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_55: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_59: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_71: init type = call %IntLiteral.ref.loc4_59() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.1: type = value_of_initializer %int_literal.make_type.loc4_71 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.2: type = converted %int_literal.make_type.loc4_71, %.loc4_71.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_11: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_15: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_27: init type = call %IntLiteral.ref.loc4_15() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.2: type = value_of_initializer %int_literal.make_type.loc4_27 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.3: type = converted %int_literal.make_type.loc4_27, %.loc4_27.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_49.1: type = splice_block %.loc4_49.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_33: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_37: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_49: init type = call %IntLiteral.ref.loc4_37() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.2: type = value_of_initializer %int_literal.make_type.loc4_49 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.3: type = converted %int_literal.make_type.loc4_49, %.loc4_49.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_34.1: type = splice_block %.loc6_34.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.3: type = converted %int_literal.make_type, %.loc6_34.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %Expect.loc7_43.1: type = class_type @Expect, @Expect(constants.%N) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_29.1: type = splice_block %.loc7_29.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.3: type = converted %int_literal.make_type, %.loc7_29.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_43.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.smul";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_9.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   %Expect.loc7_43.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_43.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_43.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral) -> %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_55.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_55.2: init @Test.%Expect.loc7_43.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_56: init @Test.%Expect.loc7_43.2 (%Expect.1) = converted %.loc7_55.1, %.loc7_55.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_56 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc10: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_0.loc10_12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc10_15: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.smul.loc10: init Core.IntLiteral = call %Mul.ref.loc10(%int_0.loc10_12, %int_0.loc10_15) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.1: Core.IntLiteral = value_of_initializer %int.smul.loc10 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.2: Core.IntLiteral = converted %int.smul.loc10, %.loc10_17.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc11: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_0.loc11_12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_3.loc11: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int.smul.loc11: init Core.IntLiteral = call %Mul.ref.loc11(%int_0.loc11_12, %int_3.loc11) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_17.1: Core.IntLiteral = value_of_initializer %int.smul.loc11 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_17.2: Core.IntLiteral = converted %int.smul.loc11, %.loc11_17.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_17.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11_29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_17.4: ref %Expect.2 = temporary %.loc11_17.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc12: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc12_15: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.smul.loc12: init Core.IntLiteral = call %Mul.ref.loc12(%int_1.loc12, %int_2.loc12_15) [template = constants.%int_2]
// CHECK:STDOUT:   %.loc12_17.1: Core.IntLiteral = value_of_initializer %int.smul.loc12 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc12_17.2: Core.IntLiteral = converted %int.smul.loc12, %.loc12_17.1 [template = constants.%int_2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_17.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc12_29: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_17.4: ref %Expect.3 = temporary %.loc12_17.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc13: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_3.loc13: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.smul.loc13: init Core.IntLiteral = call %Mul.ref.loc13(%int_3.loc13, %int_2.loc13) [template = constants.%int_6]
// CHECK:STDOUT:   %.loc13_17.1: Core.IntLiteral = value_of_initializer %int.smul.loc13 [template = constants.%int_6]
// CHECK:STDOUT:   %.loc13_17.2: Core.IntLiteral = converted %int.smul.loc13, %.loc13_17.1 [template = constants.%int_6]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_6) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc13_17.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.4 = call %Test.specific_fn.loc13() to %.loc13_17.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_6) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc13_17.4: ref %Expect.4 = temporary %.loc13_17.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc14: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_2147483647.loc14_12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.smul.loc14: init Core.IntLiteral = call %Mul.ref.loc14(%int_2147483647.loc14_12, %int_1.loc14) [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc14_27.1: Core.IntLiteral = value_of_initializer %int.smul.loc14 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc14_27.2: Core.IntLiteral = converted %int.smul.loc14, %.loc14_27.1 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_2147483647) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc14_27.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.5 = call %Test.specific_fn.loc14() to %.loc14_27.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc14_39: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_2147483647) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc14_27.4: ref %Expect.5 = temporary %.loc14_27.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc15: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_2147483647.loc15_12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15_25: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_25: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_25 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_25: init Core.IntLiteral = call %Op.bound.loc15_25(%int_1.loc15) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_25.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_25 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_25.2: Core.IntLiteral = converted %int.snegate.loc15_25, %.loc15_25.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.smul.loc15: init Core.IntLiteral = call %Mul.ref.loc15(%int_2147483647.loc15_12, %.loc15_25.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc15_28.1: Core.IntLiteral = value_of_initializer %int.smul.loc15 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc15_28.2: Core.IntLiteral = converted %int.smul.loc15, %.loc15_28.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-2147483647) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc15_28.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.6 = call %Test.specific_fn.loc15() to %.loc15_28.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc15_41: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc15_40: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_40: <bound method> = bound_method %int_2147483647.loc15_41, %impl.elem0.loc15_40 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc15_40: init Core.IntLiteral = call %Op.bound.loc15_40(%int_2147483647.loc15_41) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc15_52.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_40 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc15_52.2: Core.IntLiteral = converted %int.snegate.loc15_40, %.loc15_52.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-2147483647) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc15_28.4: ref %Expect.6 = temporary %.loc15_28.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc18: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc18: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc18_12: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.smul.loc18: init Core.IntLiteral = call %Mul.ref.loc18(%int_9223372036854775808.loc18_12, %int_1.loc18) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc18_37.1: Core.IntLiteral = value_of_initializer %int.smul.loc18 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc18_37.2: Core.IntLiteral = converted %int.smul.loc18, %.loc18_37.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc18: <specific function> = specific_function %Test.ref.loc18, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc18_37.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc18: init %Expect.7 = call %Test.specific_fn.loc18() to %.loc18_37.3
// CHECK:STDOUT:   %Expect.ref.loc18: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc18_49: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc18: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc18_37.4: ref %Expect.7 = temporary %.loc18_37.3, %Test.call.loc18
// CHECK:STDOUT:   %Test.ref.loc19: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc19: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc19_12: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc19_35: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_35: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_35 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc19_35: init Core.IntLiteral = call %Op.bound.loc19_35(%int_1.loc19) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc19_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_35 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc19_35.2: Core.IntLiteral = converted %int.snegate.loc19_35, %.loc19_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.smul.loc19: init Core.IntLiteral = call %Mul.ref.loc19(%int_9223372036854775808.loc19_12, %.loc19_35.2) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_38.1: Core.IntLiteral = value_of_initializer %int.smul.loc19 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_38.2: Core.IntLiteral = converted %int.smul.loc19, %.loc19_38.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc19: <specific function> = specific_function %Test.ref.loc19, @Test(constants.%int_-9223372036854775808) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc19_38.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc19: init %Expect.8 = call %Test.specific_fn.loc19() to %.loc19_38.3
// CHECK:STDOUT:   %Expect.ref.loc19: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc19_51: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc19_50: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_50: <bound method> = bound_method %int_9223372036854775808.loc19_51, %impl.elem0.loc19_50 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc19_50: init Core.IntLiteral = call %Op.bound.loc19_50(%int_9223372036854775808.loc19_51) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_72.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_50 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_72.2: Core.IntLiteral = converted %int.snegate.loc19_50, %.loc19_72.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Expect.loc19: type = class_type @Expect, @Expect(constants.%int_-9223372036854775808) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc19_38.4: ref %Expect.8 = temporary %.loc19_38.3, %Test.call.loc19
// CHECK:STDOUT:   %Test.ref.loc20: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc20: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc20_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc20_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc20_12: <bound method> = bound_method %int_9223372036854775808.loc20_13, %impl.elem0.loc20_12 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc20_12: init Core.IntLiteral = call %Op.bound.loc20_12(%int_9223372036854775808.loc20_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc20_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc20_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc20_12.2: Core.IntLiteral = converted %int.snegate.loc20_12, %.loc20_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.smul.loc20: init Core.IntLiteral = call %Mul.ref.loc20(%.loc20_12.2, %int_1.loc20) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc20_38.1: Core.IntLiteral = value_of_initializer %int.smul.loc20 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc20_38.2: Core.IntLiteral = converted %int.smul.loc20, %.loc20_38.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc20: <specific function> = specific_function %Test.ref.loc20, @Test(constants.%int_-9223372036854775808) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc20_38.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc20: init %Expect.8 = call %Test.specific_fn.loc20() to %.loc20_38.3
// CHECK:STDOUT:   %Expect.ref.loc20: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc20_51: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc20_50: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc20_50: <bound method> = bound_method %int_9223372036854775808.loc20_51, %impl.elem0.loc20_50 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc20_50: init Core.IntLiteral = call %Op.bound.loc20_50(%int_9223372036854775808.loc20_51) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc20_72.1: Core.IntLiteral = value_of_initializer %int.snegate.loc20_50 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc20_72.2: Core.IntLiteral = converted %int.snegate.loc20_50, %.loc20_72.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Expect.loc20: type = class_type @Expect, @Expect(constants.%int_-9223372036854775808) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc20_38.4: ref %Expect.8 = temporary %.loc20_38.3, %Test.call.loc20
// CHECK:STDOUT:   %Test.ref.loc21: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc21: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc21_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc21_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc21_12: <bound method> = bound_method %int_9223372036854775808.loc21_13, %impl.elem0.loc21_12 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc21_12: init Core.IntLiteral = call %Op.bound.loc21_12(%int_9223372036854775808.loc21_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc21_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc21_36: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_36 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc21_36: init Core.IntLiteral = call %Op.bound.loc21_36(%int_1.loc21) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc21_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc21_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc21_12.2: Core.IntLiteral = converted %int.snegate.loc21_12, %.loc21_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc21_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc21_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc21_36.2: Core.IntLiteral = converted %int.snegate.loc21_36, %.loc21_36.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.smul.loc21: init Core.IntLiteral = call %Mul.ref.loc21(%.loc21_12.2, %.loc21_36.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc21_39.1: Core.IntLiteral = value_of_initializer %int.smul.loc21 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc21_39.2: Core.IntLiteral = converted %int.smul.loc21, %.loc21_39.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc21: <specific function> = specific_function %Test.ref.loc21, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc21_39.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc21: init %Expect.7 = call %Test.specific_fn.loc21() to %.loc21_39.3
// CHECK:STDOUT:   %Expect.ref.loc21: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc21_51: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc21: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc21_39.4: ref %Expect.7 = temporary %.loc21_39.3, %Test.call.loc21
// CHECK:STDOUT:   %Test.ref.loc22: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc22: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc22_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc22_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc22_12: <bound method> = bound_method %int_9223372036854775808.loc22_13, %impl.elem0.loc22_12 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc22_12: init Core.IntLiteral = call %Op.bound.loc22_12(%int_9223372036854775808.loc22_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc22_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc22_36: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_36 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc22_36: init Core.IntLiteral = call %Op.bound.loc22_36(%int_1.loc22) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc22_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc22_12.2: Core.IntLiteral = converted %int.snegate.loc22_12, %.loc22_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc22_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc22_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_36.2: Core.IntLiteral = converted %int.snegate.loc22_36, %.loc22_36.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.smul.loc22: init Core.IntLiteral = call %Mul.ref.loc22(%.loc22_12.2, %.loc22_36.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc22_39.1: Core.IntLiteral = value_of_initializer %int.smul.loc22 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc22_39.2: Core.IntLiteral = converted %int.smul.loc22, %.loc22_39.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc22: <specific function> = specific_function %Test.ref.loc22, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc22_39.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc22: init %Expect.7 = call %Test.specific_fn.loc22() to %.loc22_39.3
// CHECK:STDOUT:   %Expect.ref.loc22: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc22_51: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc22: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc22_39.4: ref %Expect.7 = temporary %.loc22_39.3, %Test.call.loc22
// CHECK:STDOUT:   %Test.ref.loc23: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Mul.ref.loc23: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_9223372036854775808.loc23_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc23_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc23_12: <bound method> = bound_method %int_9223372036854775808.loc23_13, %impl.elem0.loc23_12 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc23_12: init Core.IntLiteral = call %Op.bound.loc23_12(%int_9223372036854775808.loc23_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_9223372036854775808.loc23_37: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc23_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc23_36: <bound method> = bound_method %int_9223372036854775808.loc23_37, %impl.elem0.loc23_36 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc23_36: init Core.IntLiteral = call %Op.bound.loc23_36(%int_9223372036854775808.loc23_37) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc23_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc23_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc23_12.2: Core.IntLiteral = converted %int.snegate.loc23_12, %.loc23_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc23_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc23_36 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc23_36.2: Core.IntLiteral = converted %int.snegate.loc23_36, %.loc23_36.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.smul.loc23: init Core.IntLiteral = call %Mul.ref.loc23(%.loc23_12.2, %.loc23_36.2) [template = constants.%int_85070591730234615865843651857942052864]
// CHECK:STDOUT:   %.loc23_59.1: Core.IntLiteral = value_of_initializer %int.smul.loc23 [template = constants.%int_85070591730234615865843651857942052864]
// CHECK:STDOUT:   %.loc23_59.2: Core.IntLiteral = converted %int.smul.loc23, %.loc23_59.1 [template = constants.%int_85070591730234615865843651857942052864]
// CHECK:STDOUT:   %Test.specific_fn.loc23: <specific function> = specific_function %Test.ref.loc23, @Test(constants.%int_85070591730234615865843651857942052864) [template = constants.%Test.specific_fn.8]
// CHECK:STDOUT:   %.loc23_59.3: ref %Expect.9 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc23: init %Expect.9 = call %Test.specific_fn.loc23() to %.loc23_59.3
// CHECK:STDOUT:   %Expect.ref.loc24: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_85070591730234615865843651857942052864: Core.IntLiteral = int_value 85070591730234615865843651857942052864 [template = constants.%int_85070591730234615865843651857942052864]
// CHECK:STDOUT:   %Expect.loc24: type = class_type @Expect, @Expect(constants.%int_85070591730234615865843651857942052864) [template = constants.%Expect.9]
// CHECK:STDOUT:   %.loc23_59.4: ref %Expect.9 = temporary %.loc23_59.3, %Test.call.loc23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_6) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_6
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_6
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_6) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_6
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2147483647) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-2147483647
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-2147483647
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2147483647) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-2147483647
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-2147483647
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_85070591730234615865843651857942052864) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_85070591730234615865843651857942052864
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_85070591730234615865843651857942052864
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_85070591730234615865843651857942052864) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_85070591730234615865843651857942052864
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_85070591730234615865843651857942052864
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mul.type.1: type = fn_type @Mul.1 [template]
// CHECK:STDOUT:   %Mul: %Mul.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_32767.1: Core.IntLiteral = int_value 32767 [template]
// CHECK:STDOUT:   %int_65536.1: Core.IntLiteral = int_value 65536 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_32767.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_32767.2: %i32 = int_value 32767 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_65536.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_65536.2: %i32 = int_value 65536 [template]
// CHECK:STDOUT:   %int_2147418112: %i32 = int_value 2147418112 [template]
// CHECK:STDOUT:   %int_32768.1: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32768.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_32768.2: %i32 = int_value 32768 [template]
// CHECK:STDOUT:   %int_-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type.1 = fn_decl @Mul.1 [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_11: type = splice_block %i32.loc4_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_19: type = splice_block %i32.loc4_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref.loc6: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [template = constants.%int_32767.1]
// CHECK:STDOUT:   %int_65536.loc6: Core.IntLiteral = int_value 65536 [template = constants.%int_65536.1]
// CHECK:STDOUT:   %impl.elem0.loc6_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_18: <bound method> = bound_method %int_32767, %impl.elem0.loc6_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_18: <specific function> = specific_function %Convert.bound.loc6_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init %i32 = call %Convert.specific_fn.loc6_18(%int_32767) [template = constants.%int_32767.2]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %int.convert_checked.loc6_18 [template = constants.%int_32767.2]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_32767, %.loc6_18.1 [template = constants.%int_32767.2]
// CHECK:STDOUT:   %impl.elem0.loc6_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_26: <bound method> = bound_method %int_65536.loc6, %impl.elem0.loc6_26 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_26: <specific function> = specific_function %Convert.bound.loc6_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_26: init %i32 = call %Convert.specific_fn.loc6_26(%int_65536.loc6) [template = constants.%int_65536.2]
// CHECK:STDOUT:   %.loc6_26.1: %i32 = value_of_initializer %int.convert_checked.loc6_26 [template = constants.%int_65536.2]
// CHECK:STDOUT:   %.loc6_26.2: %i32 = converted %int_65536.loc6, %.loc6_26.1 [template = constants.%int_65536.2]
// CHECK:STDOUT:   %int.smul.loc6: init %i32 = call %Mul.ref.loc6(%.loc6_18.2, %.loc6_26.2) [template = constants.%int_2147418112]
// CHECK:STDOUT:   %.loc6_34.1: %i32 = value_of_initializer %int.smul.loc6 [template = constants.%int_2147418112]
// CHECK:STDOUT:   %.loc6_34.2: %i32 = converted %int.smul.loc6, %.loc6_34.1 [template = constants.%int_2147418112]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc6_34.2
// CHECK:STDOUT:   %Mul.ref.loc10: %Mul.type.1 = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template = constants.%int_32768.1]
// CHECK:STDOUT:   %int_65536.loc10: Core.IntLiteral = int_value 65536 [template = constants.%int_65536.1]
// CHECK:STDOUT:   %impl.elem0.loc10_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_18: <bound method> = bound_method %int_32768, %impl.elem0.loc10_18 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_18: <specific function> = specific_function %Convert.bound.loc10_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %i32 = call %Convert.specific_fn.loc10_18(%int_32768) [template = constants.%int_32768.2]
// CHECK:STDOUT:   %.loc10_18.1: %i32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%int_32768.2]
// CHECK:STDOUT:   %.loc10_18.2: %i32 = converted %int_32768, %.loc10_18.1 [template = constants.%int_32768.2]
// CHECK:STDOUT:   %impl.elem0.loc10_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_26: <bound method> = bound_method %int_65536.loc10, %impl.elem0.loc10_26 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26: <specific function> = specific_function %Convert.bound.loc10_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_26: init %i32 = call %Convert.specific_fn.loc10_26(%int_65536.loc10) [template = constants.%int_65536.2]
// CHECK:STDOUT:   %.loc10_26.1: %i32 = value_of_initializer %int.convert_checked.loc10_26 [template = constants.%int_65536.2]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = converted %int_65536.loc10, %.loc10_26.1 [template = constants.%int_65536.2]
// CHECK:STDOUT:   %int.smul.loc10: init %i32 = call %Mul.ref.loc10(%.loc10_18.2, %.loc10_26.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %int.smul.loc10 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int.smul.loc10, %.loc10_34.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc10_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
