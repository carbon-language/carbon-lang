// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(-1, 1)) as Expect(-1);
  Test(RightShift(-2, 1)) as Expect(-1);
  Test(RightShift(-10, 2)) as Expect(-3);
}

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return RightShift(a, b);
}

// --- u32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
}

fn RuntimeCall(a: u32, b: i32) -> u32 {
  return RightShift(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.right_shift";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(-1, 1)) as Expect(-1);
  Test(RightShift(-2, 1)) as Expect(-1);
  Test(RightShift(-10, 2)) as Expect(-3);
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);

  Test(RightShift(0x1234_5678, 1_000_000_000)) as Expect(0);
  Test(RightShift(-0x1234_5678, 1_000_000_000)) as Expect(-1);
  Test(RightShift(0xFFFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  Test(RightShift(0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  Test(RightShift(-0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(-1);
  Test(RightShift(-0x8000_0000_0000_0000, 1_000_000_000)) as Expect(-1);
}

// --- fail_bad_shift.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn RightShiftLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.right_shift";

// Shift greater than size is disallowed for sized types.
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 >> 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 >> 33` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either, even for literals, even if the lhs is zero.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:21: error: shift distance >= type width of 32 in `1 >> -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = RightShift(1, -1);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative: i32 = RightShift(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance >= type width of 32 in `0 >> -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:39: error: shift distance negative in `1 >> -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:44: error: shift distance negative in `0 >> -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~~
let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1.2) [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_22.1: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_22.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_22.2: %i32 = int_value 22 [template]
// CHECK:STDOUT:   %int_5.1: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_5.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_5.1) [template]
// CHECK:STDOUT:   %int_5.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_5.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1.1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_-1.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_-1.2) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2.1, %Op.2 [template]
// CHECK:STDOUT:   %int_-2.1: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_-2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-2.2: %i32 = int_value -2 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_10, %Op.2 [template]
// CHECK:STDOUT:   %int_-10.1: Core.IntLiteral = int_value -10 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_-10.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-10.2: %i32 = int_value -10 [template]
// CHECK:STDOUT:   %int_-3.1: %i32 = int_value -3 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_-3.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_-3.1) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_3, %Op.2 [template]
// CHECK:STDOUT:   %int_-3.2: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %Convert.bound.10: <bound method> = bound_method %int_-3.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.10: <specific function> = specific_function %Convert.bound.10, @Convert.2(%int_32) [template]
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
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc4_14.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_14.1, runtime_param<invalid> [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_14.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc5_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc5_9.1, runtime_param<invalid> [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc5_9.1 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc5_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc5_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc5_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT:     %int_32.loc20_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc20_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc20_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc4_14.1: %i32) {
// CHECK:STDOUT:   %N.loc4_14.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc4_14.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc5_9.1: %i32) {
// CHECK:STDOUT:   %N.loc5_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc5_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc5_29.2: type = class_type @Expect, @Expect(%N.loc5_9.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc5_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc5_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) -> %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_41.2: init @Test.%Expect.loc5_29.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc5_42: init @Test.%Expect.loc5_29.2 (%Expect.1) = converted %.loc5_41.1, %.loc5_41.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc5_42 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_0.loc10_19: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_31.loc10: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc10_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_19: <bound method> = bound_method %int_0.loc10_19, %impl.elem0.loc10_19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_19: <specific function> = specific_function %Convert.bound.loc10_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_19: init %i32 = call %Convert.specific_fn.loc10_19(%int_0.loc10_19) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.convert_checked.loc10_19 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int_0.loc10_19, %.loc10_19.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_22: <bound method> = bound_method %int_31.loc10, %impl.elem0.loc10_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_22: <specific function> = specific_function %Convert.bound.loc10_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_22: init %i32 = call %Convert.specific_fn.loc10_22(%int_31.loc10) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %int.convert_checked.loc10_22 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_31.loc10, %.loc10_22.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc10: init %i32 = call %RightShift.ref.loc10(%.loc10_19.2, %.loc10_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %int.right_shift.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %int.right_shift.loc10, %.loc10_25.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_25.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_38: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10_38 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_38: <specific function> = specific_function %Convert.bound.loc10_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_38: init %i32 = call %Convert.specific_fn.loc10_38(%int_0.loc10_37) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_38.1: %i32 = value_of_initializer %int.convert_checked.loc10_38 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_38.2: %i32 = converted %int_0.loc10_37, %.loc10_38.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_25.4: ref %Expect.2 = temporary %.loc10_25.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc11: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31.loc11: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc11_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_19: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_19: <specific function> = specific_function %Convert.bound.loc11_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_19: init %i32 = call %Convert.specific_fn.loc11_19(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.convert_checked.loc11_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_1.loc11, %.loc11_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_22: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_22: <bound method> = bound_method %int_31.loc11, %impl.elem0.loc11_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_22: <specific function> = specific_function %Convert.bound.loc11_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_22: init %i32 = call %Convert.specific_fn.loc11_22(%int_31.loc11) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc11_22.1: %i32 = value_of_initializer %int.convert_checked.loc11_22 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc11_22.2: %i32 = converted %int_31.loc11, %.loc11_22.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc11: init %i32 = call %RightShift.ref.loc11(%.loc11_19.2, %.loc11_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_25.1: %i32 = value_of_initializer %int.right_shift.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_25.2: %i32 = converted %int.right_shift.loc11, %.loc11_25.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_25.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_38: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11_38 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_38: <specific function> = specific_function %Convert.bound.loc11_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_38: init %i32 = call %Convert.specific_fn.loc11_38(%int_0.loc11) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_38.1: %i32 = value_of_initializer %int.convert_checked.loc11_38 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_38.2: %i32 = converted %int_0.loc11, %.loc11_38.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_25.4: ref %Expect.2 = temporary %.loc11_25.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc12: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc12_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc12_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_19: <bound method> = bound_method %int_1.loc12_19, %impl.elem0.loc12_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_19: <specific function> = specific_function %Convert.bound.loc12_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_19: init %i32 = call %Convert.specific_fn.loc12_19(%int_1.loc12_19) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %int.convert_checked.loc12_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int_1.loc12_19, %.loc12_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_22: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_22: <bound method> = bound_method %int_0.loc12, %impl.elem0.loc12_22 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_22: <specific function> = specific_function %Convert.bound.loc12_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_22: init %i32 = call %Convert.specific_fn.loc12_22(%int_0.loc12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %int.convert_checked.loc12_22 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %int_0.loc12, %.loc12_22.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.right_shift.loc12: init %i32 = call %RightShift.ref.loc12(%.loc12_19.2, %.loc12_22.2) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_24.1: %i32 = value_of_initializer %int.right_shift.loc12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_24.2: %i32 = converted %int.right_shift.loc12, %.loc12_24.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_1.2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_24.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc12_36: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_37: <bound method> = bound_method %int_1.loc12_36, %impl.elem0.loc12_37 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_37: <specific function> = specific_function %Convert.bound.loc12_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_37: init %i32 = call %Convert.specific_fn.loc12_37(%int_1.loc12_36) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %int.convert_checked.loc12_37 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_1.loc12_36, %.loc12_37.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_1.2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_24.4: ref %Expect.3 = temporary %.loc12_24.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc13_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_19: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_19: <specific function> = specific_function %Convert.bound.loc13_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_19: init %i32 = call %Convert.specific_fn.loc13_19(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_19.1: %i32 = value_of_initializer %int.convert_checked.loc13_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_19.2: %i32 = converted %int_1.loc13, %.loc13_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_22: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_22: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_22 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_22: <specific function> = specific_function %Convert.bound.loc13_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc13_22: init %i32 = call %Convert.specific_fn.loc13_22(%int_2.loc13) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_22.1: %i32 = value_of_initializer %int.convert_checked.loc13_22 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_22.2: %i32 = converted %int_2.loc13, %.loc13_22.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc13: init %i32 = call %RightShift.ref.loc13(%.loc13_19.2, %.loc13_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.1: %i32 = value_of_initializer %int.right_shift.loc13 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.2: %i32 = converted %int.right_shift.loc13, %.loc13_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_24.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc13_37: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_37: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13_37 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_37: <specific function> = specific_function %Convert.bound.loc13_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_37: init %i32 = call %Convert.specific_fn.loc13_37(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_37.1: %i32 = value_of_initializer %int.convert_checked.loc13_37 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_37.2: %i32 = converted %int_0.loc13, %.loc13_37.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_24.4: ref %Expect.2 = temporary %.loc13_24.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template = constants.%int_22.1]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc14_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_19: <bound method> = bound_method %int_22, %impl.elem0.loc14_19 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_19: <specific function> = specific_function %Convert.bound.loc14_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc14_19: init %i32 = call %Convert.specific_fn.loc14_19(%int_22) [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc14_19.1: %i32 = value_of_initializer %int.convert_checked.loc14_19 [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc14_19.2: %i32 = converted %int_22, %.loc14_19.1 [template = constants.%int_22.2]
// CHECK:STDOUT:   %impl.elem0.loc14_23: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_23: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_23 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_23: <specific function> = specific_function %Convert.bound.loc14_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_23: init %i32 = call %Convert.specific_fn.loc14_23(%int_2.loc14) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_23.1: %i32 = value_of_initializer %int.convert_checked.loc14_23 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_23.2: %i32 = converted %int_2.loc14, %.loc14_23.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc14: init %i32 = call %RightShift.ref.loc14(%.loc14_19.2, %.loc14_23.2) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_25.1: %i32 = value_of_initializer %int.right_shift.loc14 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_25.2: %i32 = converted %int.right_shift.loc14, %.loc14_25.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_5.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_25.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_25.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc14_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_38: <bound method> = bound_method %int_5, %impl.elem0.loc14_38 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_38: <specific function> = specific_function %Convert.bound.loc14_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc14_38: init %i32 = call %Convert.specific_fn.loc14_38(%int_5) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_38.1: %i32 = value_of_initializer %int.convert_checked.loc14_38 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_38.2: %i32 = converted %int_5, %.loc14_38.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_5.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_25.4: ref %Expect.4 = temporary %.loc14_25.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc15: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc15_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_19.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_19: <bound method> = bound_method %int_1.loc15_20, %impl.elem0.loc15_19.1 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_19: init Core.IntLiteral = call %Op.bound.loc15_19(%int_1.loc15_20) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc15_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_19.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_19: <bound method> = bound_method %int.snegate.loc15_19, %impl.elem0.loc15_19.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_19: <specific function> = specific_function %Convert.bound.loc15_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc15_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_19 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_19.2: Core.IntLiteral = converted %int.snegate.loc15_19, %.loc15_19.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_19: init %i32 = call %Convert.specific_fn.loc15_19(%.loc15_19.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_19.3: %i32 = value_of_initializer %int.convert_checked.loc15_19 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_19.4: %i32 = converted %int.snegate.loc15_19, %.loc15_19.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %impl.elem0.loc15_23: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_23: <bound method> = bound_method %int_1.loc15_23, %impl.elem0.loc15_23 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_23: <specific function> = specific_function %Convert.bound.loc15_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_23: init %i32 = call %Convert.specific_fn.loc15_23(%int_1.loc15_23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_23.1: %i32 = value_of_initializer %int.convert_checked.loc15_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_23.2: %i32 = converted %int_1.loc15_23, %.loc15_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.right_shift.loc15: init %i32 = call %RightShift.ref.loc15(%.loc15_19.4, %.loc15_23.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_25.1: %i32 = value_of_initializer %int.right_shift.loc15 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_25.2: %i32 = converted %int.right_shift.loc15, %.loc15_25.1 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-1.2) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_25.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_25.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc15_38: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_37: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_37: <bound method> = bound_method %int_1.loc15_38, %impl.elem0.loc15_37 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_37: init Core.IntLiteral = call %Op.bound.loc15_37(%int_1.loc15_38) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_39: <bound method> = bound_method %int.snegate.loc15_37, %impl.elem0.loc15_39 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_39: <specific function> = specific_function %Convert.bound.loc15_39, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc15_37.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_37 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_37.2: Core.IntLiteral = converted %int.snegate.loc15_37, %.loc15_37.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_39: init %i32 = call %Convert.specific_fn.loc15_39(%.loc15_37.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_39.1: %i32 = value_of_initializer %int.convert_checked.loc15_39 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_39.2: %i32 = converted %int.snegate.loc15_37, %.loc15_39.1 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-1.2) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_25.4: ref %Expect.5 = temporary %.loc15_25.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc16: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc16_19.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16_19: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_19.1 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc16_19: init Core.IntLiteral = call %Op.bound.loc16_19(%int_2.loc16) [template = constants.%int_-2.1]
// CHECK:STDOUT:   %int_1.loc16_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_19.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_19: <bound method> = bound_method %int.snegate.loc16_19, %impl.elem0.loc16_19.2 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_19: <specific function> = specific_function %Convert.bound.loc16_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %.loc16_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_19 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc16_19.2: Core.IntLiteral = converted %int.snegate.loc16_19, %.loc16_19.1 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_19: init %i32 = call %Convert.specific_fn.loc16_19(%.loc16_19.2) [template = constants.%int_-2.2]
// CHECK:STDOUT:   %.loc16_19.3: %i32 = value_of_initializer %int.convert_checked.loc16_19 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %.loc16_19.4: %i32 = converted %int.snegate.loc16_19, %.loc16_19.3 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %impl.elem0.loc16_23: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_23: <bound method> = bound_method %int_1.loc16_23, %impl.elem0.loc16_23 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_23: <specific function> = specific_function %Convert.bound.loc16_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc16_23: init %i32 = call %Convert.specific_fn.loc16_23(%int_1.loc16_23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_23.1: %i32 = value_of_initializer %int.convert_checked.loc16_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_23.2: %i32 = converted %int_1.loc16_23, %.loc16_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.right_shift.loc16: init %i32 = call %RightShift.ref.loc16(%.loc16_19.4, %.loc16_23.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %int.right_shift.loc16 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %int.right_shift.loc16, %.loc16_25.1 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_-1.2) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc16_25.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.5 = call %Test.specific_fn.loc16() to %.loc16_25.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc16_38: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_37: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16_37: <bound method> = bound_method %int_1.loc16_38, %impl.elem0.loc16_37 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc16_37: init Core.IntLiteral = call %Op.bound.loc16_37(%int_1.loc16_38) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_39: <bound method> = bound_method %int.snegate.loc16_37, %impl.elem0.loc16_39 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_39: <specific function> = specific_function %Convert.bound.loc16_39, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc16_37.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_37 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc16_37.2: Core.IntLiteral = converted %int.snegate.loc16_37, %.loc16_37.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_39: init %i32 = call %Convert.specific_fn.loc16_39(%.loc16_37.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_39.1: %i32 = value_of_initializer %int.convert_checked.loc16_39 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_39.2: %i32 = converted %int.snegate.loc16_37, %.loc16_39.1 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_-1.2) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc16_25.4: ref %Expect.5 = temporary %.loc16_25.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc17: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc17_19.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc17_19: <bound method> = bound_method %int_10, %impl.elem0.loc17_19.1 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc17_19: init Core.IntLiteral = call %Op.bound.loc17_19(%int_10) [template = constants.%int_-10.1]
// CHECK:STDOUT:   %int_2.loc17: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc17_19.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_19: <bound method> = bound_method %int.snegate.loc17_19, %impl.elem0.loc17_19.2 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_19: <specific function> = specific_function %Convert.bound.loc17_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %.loc17_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_19 [template = constants.%int_-10.1]
// CHECK:STDOUT:   %.loc17_19.2: Core.IntLiteral = converted %int.snegate.loc17_19, %.loc17_19.1 [template = constants.%int_-10.1]
// CHECK:STDOUT:   %int.convert_checked.loc17_19: init %i32 = call %Convert.specific_fn.loc17_19(%.loc17_19.2) [template = constants.%int_-10.2]
// CHECK:STDOUT:   %.loc17_19.3: %i32 = value_of_initializer %int.convert_checked.loc17_19 [template = constants.%int_-10.2]
// CHECK:STDOUT:   %.loc17_19.4: %i32 = converted %int.snegate.loc17_19, %.loc17_19.3 [template = constants.%int_-10.2]
// CHECK:STDOUT:   %impl.elem0.loc17_24: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_24: <bound method> = bound_method %int_2.loc17, %impl.elem0.loc17_24 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_24: <specific function> = specific_function %Convert.bound.loc17_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc17_24: init %i32 = call %Convert.specific_fn.loc17_24(%int_2.loc17) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc17_24.1: %i32 = value_of_initializer %int.convert_checked.loc17_24 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc17_24.2: %i32 = converted %int_2.loc17, %.loc17_24.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc17: init %i32 = call %RightShift.ref.loc17(%.loc17_19.4, %.loc17_24.2) [template = constants.%int_-3.1]
// CHECK:STDOUT:   %.loc17_26.1: %i32 = value_of_initializer %int.right_shift.loc17 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %.loc17_26.2: %i32 = converted %int.right_shift.loc17, %.loc17_26.1 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_-3.1) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc17_26.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.6 = call %Test.specific_fn.loc17() to %.loc17_26.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc17_38: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc17_38: <bound method> = bound_method %int_3, %impl.elem0.loc17_38 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc17_38: init Core.IntLiteral = call %Op.bound.loc17_38(%int_3) [template = constants.%int_-3.2]
// CHECK:STDOUT:   %impl.elem0.loc17_40: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_40: <bound method> = bound_method %int.snegate.loc17_38, %impl.elem0.loc17_40 [template = constants.%Convert.bound.10]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_40: <specific function> = specific_function %Convert.bound.loc17_40, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.10]
// CHECK:STDOUT:   %.loc17_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_38 [template = constants.%int_-3.2]
// CHECK:STDOUT:   %.loc17_38.2: Core.IntLiteral = converted %int.snegate.loc17_38, %.loc17_38.1 [template = constants.%int_-3.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_40: init %i32 = call %Convert.specific_fn.loc17_40(%.loc17_38.2) [template = constants.%int_-3.1]
// CHECK:STDOUT:   %.loc17_40.1: %i32 = value_of_initializer %int.convert_checked.loc17_40 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %.loc17_40.2: %i32 = converted %int.snegate.loc17_38, %.loc17_40.1 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_-3.1) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc17_26.4: ref %Expect.6 = temporary %.loc17_26.3, %Test.call.loc17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc21_26.1: %i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc21_26.2: %i32 = converted %int.right_shift, %.loc21_26.1
// CHECK:STDOUT:   return %.loc21_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_1.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_5.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_5.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_5.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_5.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_5.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_5.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-1.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-1.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_-1.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-1.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_-1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-3.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-3.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_-3.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-3.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-3.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_-3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %N.2: %u32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %u32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%u32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %u32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.7, @impl.5(%int_32) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %u32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_0.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_0.3: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1.2) [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_2.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_22.1: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_22.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_22.2: %u32 = int_value 22 [template]
// CHECK:STDOUT:   %int_5.1: %u32 = int_value 5 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_5.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_5.1) [template]
// CHECK:STDOUT:   %int_5.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_5.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4294967295.1: Core.IntLiteral = int_value 4294967295 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_4294967295.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4294967295.2: %u32 = int_value 4294967295 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_1.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_1.3: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2147483647.1: %u32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_2147483647.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_2147483647.1) [template]
// CHECK:STDOUT:   %int_2147483647.2: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.10: <bound method> = bound_method %int_2147483647.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.10: <specific function> = specific_function %Convert.bound.10, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2882400001.1: Core.IntLiteral = int_value 2882400001 [template]
// CHECK:STDOUT:   %int_8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.11: <bound method> = bound_method %int_2882400001.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.11: <specific function> = specific_function %Convert.bound.11, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2882400001.2: %u32 = int_value 2882400001 [template]
// CHECK:STDOUT:   %Convert.bound.12: <bound method> = bound_method %int_8.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.12: <specific function> = specific_function %Convert.bound.12, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %int_11259375.1: %u32 = int_value 11259375 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_11259375.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_11259375.1) [template]
// CHECK:STDOUT:   %int_11259375.2: Core.IntLiteral = int_value 11259375 [template]
// CHECK:STDOUT:   %Convert.bound.13: <bound method> = bound_method %int_11259375.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.13: <specific function> = specific_function %Convert.bound.13, @Convert.2(%int_32) [template]
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
// CHECK:STDOUT:     .UInt = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc4_14.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc4_14.1, runtime_param<invalid> [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_14.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc5_9.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc5_9.1, runtime_param<invalid> [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %u32 = name_ref N, %N.loc5_9.1 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc5_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc5_9.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc5_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc7_18: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc7_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc7_34: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc19_19: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc19_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc19_35: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc4_14.1: %u32) {
// CHECK:STDOUT:   %N.loc4_14.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc4_14.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc5_9.1: %u32) {
// CHECK:STDOUT:   %N.loc5_9.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc5_9.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc5_29.2: type = class_type @Expect, @Expect(%N.loc5_9.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc5_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc5_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %u32) -> %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_41.2: init @Test.%Expect.loc5_29.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc5_42: init @Test.%Expect.loc5_29.2 (%Expect.1) = converted %.loc5_41.1, %.loc5_41.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc5_42 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %u32, %b.param_patt: %i32) -> %u32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_0.loc10_19: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_31.loc10: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc10_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_19: <bound method> = bound_method %int_0.loc10_19, %impl.elem0.loc10_19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_19: <specific function> = specific_function %Convert.bound.loc10_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_19: init %u32 = call %Convert.specific_fn.loc10_19(%int_0.loc10_19) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_19.1: %u32 = value_of_initializer %int.convert_checked.loc10_19 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_19.2: %u32 = converted %int_0.loc10_19, %.loc10_19.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc10_22: <bound method> = bound_method %int_31.loc10, %impl.elem0.loc10_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_22: <specific function> = specific_function %Convert.bound.loc10_22, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_22: init %i32 = call %Convert.specific_fn.loc10_22(%int_31.loc10) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %int.convert_checked.loc10_22 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_31.loc10, %.loc10_22.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc10: init %u32 = call %RightShift.ref.loc10(%.loc10_19.2, %.loc10_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_25.1: %u32 = value_of_initializer %int.right_shift.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_25.2: %u32 = converted %int.right_shift.loc10, %.loc10_25.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_25.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_38: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10_38 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_38: <specific function> = specific_function %Convert.bound.loc10_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_38: init %u32 = call %Convert.specific_fn.loc10_38(%int_0.loc10_37) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_38.1: %u32 = value_of_initializer %int.convert_checked.loc10_38 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_38.2: %u32 = converted %int_0.loc10_37, %.loc10_38.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_25.4: ref %Expect.2 = temporary %.loc10_25.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc11: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31.loc11: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc11_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_19: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_19: <specific function> = specific_function %Convert.bound.loc11_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_19: init %u32 = call %Convert.specific_fn.loc11_19(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_19.1: %u32 = value_of_initializer %int.convert_checked.loc11_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_19.2: %u32 = converted %int_1.loc11, %.loc11_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_22: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc11_22: <bound method> = bound_method %int_31.loc11, %impl.elem0.loc11_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_22: <specific function> = specific_function %Convert.bound.loc11_22, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_22: init %i32 = call %Convert.specific_fn.loc11_22(%int_31.loc11) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc11_22.1: %i32 = value_of_initializer %int.convert_checked.loc11_22 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc11_22.2: %i32 = converted %int_31.loc11, %.loc11_22.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc11: init %u32 = call %RightShift.ref.loc11(%.loc11_19.2, %.loc11_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_25.1: %u32 = value_of_initializer %int.right_shift.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_25.2: %u32 = converted %int.right_shift.loc11, %.loc11_25.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_25.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_38: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11_38 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_38: <specific function> = specific_function %Convert.bound.loc11_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_38: init %u32 = call %Convert.specific_fn.loc11_38(%int_0.loc11) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_38.1: %u32 = value_of_initializer %int.convert_checked.loc11_38 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_38.2: %u32 = converted %int_0.loc11, %.loc11_38.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_25.4: ref %Expect.2 = temporary %.loc11_25.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc12: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc12_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc12_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_19: <bound method> = bound_method %int_1.loc12_19, %impl.elem0.loc12_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_19: <specific function> = specific_function %Convert.bound.loc12_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_19: init %u32 = call %Convert.specific_fn.loc12_19(%int_1.loc12_19) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.1: %u32 = value_of_initializer %int.convert_checked.loc12_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.2: %u32 = converted %int_1.loc12_19, %.loc12_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_22: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc12_22: <bound method> = bound_method %int_0.loc12, %impl.elem0.loc12_22 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_22: <specific function> = specific_function %Convert.bound.loc12_22, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc12_22: init %i32 = call %Convert.specific_fn.loc12_22(%int_0.loc12) [template = constants.%int_0.3]
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %int.convert_checked.loc12_22 [template = constants.%int_0.3]
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %int_0.loc12, %.loc12_22.1 [template = constants.%int_0.3]
// CHECK:STDOUT:   %int.right_shift.loc12: init %u32 = call %RightShift.ref.loc12(%.loc12_19.2, %.loc12_22.2) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_24.1: %u32 = value_of_initializer %int.right_shift.loc12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_24.2: %u32 = converted %int.right_shift.loc12, %.loc12_24.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_1.2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_24.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc12_36: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_37: <bound method> = bound_method %int_1.loc12_36, %impl.elem0.loc12_37 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_37: <specific function> = specific_function %Convert.bound.loc12_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_37: init %u32 = call %Convert.specific_fn.loc12_37(%int_1.loc12_36) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_37.1: %u32 = value_of_initializer %int.convert_checked.loc12_37 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_37.2: %u32 = converted %int_1.loc12_36, %.loc12_37.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_1.2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_24.4: ref %Expect.3 = temporary %.loc12_24.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc13_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_19: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_19: <specific function> = specific_function %Convert.bound.loc13_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_19: init %u32 = call %Convert.specific_fn.loc13_19(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_19.1: %u32 = value_of_initializer %int.convert_checked.loc13_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_19.2: %u32 = converted %int_1.loc13, %.loc13_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_22: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc13_22: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_22 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_22: <specific function> = specific_function %Convert.bound.loc13_22, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc13_22: init %i32 = call %Convert.specific_fn.loc13_22(%int_2.loc13) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_22.1: %i32 = value_of_initializer %int.convert_checked.loc13_22 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_22.2: %i32 = converted %int_2.loc13, %.loc13_22.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc13: init %u32 = call %RightShift.ref.loc13(%.loc13_19.2, %.loc13_22.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.1: %u32 = value_of_initializer %int.right_shift.loc13 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.2: %u32 = converted %int.right_shift.loc13, %.loc13_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_24.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc13_37: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_37: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13_37 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_37: <specific function> = specific_function %Convert.bound.loc13_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_37: init %u32 = call %Convert.specific_fn.loc13_37(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_37.1: %u32 = value_of_initializer %int.convert_checked.loc13_37 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_37.2: %u32 = converted %int_0.loc13, %.loc13_37.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_24.4: ref %Expect.2 = temporary %.loc13_24.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template = constants.%int_22.1]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc14_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_19: <bound method> = bound_method %int_22, %impl.elem0.loc14_19 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_19: <specific function> = specific_function %Convert.bound.loc14_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc14_19: init %u32 = call %Convert.specific_fn.loc14_19(%int_22) [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc14_19.1: %u32 = value_of_initializer %int.convert_checked.loc14_19 [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc14_19.2: %u32 = converted %int_22, %.loc14_19.1 [template = constants.%int_22.2]
// CHECK:STDOUT:   %impl.elem0.loc14_23: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc14_23: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_23 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_23: <specific function> = specific_function %Convert.bound.loc14_23, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc14_23: init %i32 = call %Convert.specific_fn.loc14_23(%int_2.loc14) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_23.1: %i32 = value_of_initializer %int.convert_checked.loc14_23 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_23.2: %i32 = converted %int_2.loc14, %.loc14_23.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc14: init %u32 = call %RightShift.ref.loc14(%.loc14_19.2, %.loc14_23.2) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_25.1: %u32 = value_of_initializer %int.right_shift.loc14 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_25.2: %u32 = converted %int.right_shift.loc14, %.loc14_25.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_5.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_25.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_25.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc14_38: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_38: <bound method> = bound_method %int_5, %impl.elem0.loc14_38 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_38: <specific function> = specific_function %Convert.bound.loc14_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc14_38: init %u32 = call %Convert.specific_fn.loc14_38(%int_5) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_38.1: %u32 = value_of_initializer %int.convert_checked.loc14_38 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc14_38.2: %u32 = converted %int_5, %.loc14_38.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_5.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_25.4: ref %Expect.4 = temporary %.loc14_25.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc15: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template = constants.%int_4294967295.1]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_19: <bound method> = bound_method %int_4294967295, %impl.elem0.loc15_19 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_19: <specific function> = specific_function %Convert.bound.loc15_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc15_19: init %u32 = call %Convert.specific_fn.loc15_19(%int_4294967295) [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %.loc15_19.1: %u32 = value_of_initializer %int.convert_checked.loc15_19 [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %.loc15_19.2: %u32 = converted %int_4294967295, %.loc15_19.1 [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %impl.elem0.loc15_32: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc15_32: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_32 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_32: <specific function> = specific_function %Convert.bound.loc15_32, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc15_32: init %i32 = call %Convert.specific_fn.loc15_32(%int_1.loc15) [template = constants.%int_1.3]
// CHECK:STDOUT:   %.loc15_32.1: %i32 = value_of_initializer %int.convert_checked.loc15_32 [template = constants.%int_1.3]
// CHECK:STDOUT:   %.loc15_32.2: %i32 = converted %int_1.loc15, %.loc15_32.1 [template = constants.%int_1.3]
// CHECK:STDOUT:   %int.right_shift.loc15: init %u32 = call %RightShift.ref.loc15(%.loc15_19.2, %.loc15_32.2) [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc15_34.1: %u32 = value_of_initializer %int.right_shift.loc15 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc15_34.2: %u32 = converted %int.right_shift.loc15, %.loc15_34.1 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_2147483647.1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_34.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_34.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc15_57: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_57: <bound method> = bound_method %int_2147483647, %impl.elem0.loc15_57 [template = constants.%Convert.bound.10]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_57: <specific function> = specific_function %Convert.bound.loc15_57, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.10]
// CHECK:STDOUT:   %int.convert_checked.loc15_57: init %u32 = call %Convert.specific_fn.loc15_57(%int_2147483647) [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc15_57.1: %u32 = value_of_initializer %int.convert_checked.loc15_57 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc15_57.2: %u32 = converted %int_2147483647, %.loc15_57.1 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_2147483647.1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_34.4: ref %Expect.5 = temporary %.loc15_34.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc16: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template = constants.%int_2882400001.1]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %impl.elem0.loc16_19: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_19: <bound method> = bound_method %int_2882400001, %impl.elem0.loc16_19 [template = constants.%Convert.bound.11]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_19: <specific function> = specific_function %Convert.bound.loc16_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.11]
// CHECK:STDOUT:   %int.convert_checked.loc16_19: init %u32 = call %Convert.specific_fn.loc16_19(%int_2882400001) [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %.loc16_19.1: %u32 = value_of_initializer %int.convert_checked.loc16_19 [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %.loc16_19.2: %u32 = converted %int_2882400001, %.loc16_19.1 [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %impl.elem0.loc16_32: %Convert.type.11 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc16_32: <bound method> = bound_method %int_8, %impl.elem0.loc16_32 [template = constants.%Convert.bound.12]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_32: <specific function> = specific_function %Convert.bound.loc16_32, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.12]
// CHECK:STDOUT:   %int.convert_checked.loc16_32: init %i32 = call %Convert.specific_fn.loc16_32(%int_8) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc16_32.1: %i32 = value_of_initializer %int.convert_checked.loc16_32 [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc16_32.2: %i32 = converted %int_8, %.loc16_32.1 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int.right_shift.loc16: init %u32 = call %RightShift.ref.loc16(%.loc16_19.2, %.loc16_32.2) [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %.loc16_34.1: %u32 = value_of_initializer %int.right_shift.loc16 [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %.loc16_34.2: %u32 = converted %int.right_shift.loc16, %.loc16_34.1 [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_11259375.1) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc16_34.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.6 = call %Test.specific_fn.loc16() to %.loc16_34.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_11259375: Core.IntLiteral = int_value 11259375 [template = constants.%int_11259375.2]
// CHECK:STDOUT:   %impl.elem0.loc16_55: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_55: <bound method> = bound_method %int_11259375, %impl.elem0.loc16_55 [template = constants.%Convert.bound.13]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55: <specific function> = specific_function %Convert.bound.loc16_55, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.13]
// CHECK:STDOUT:   %int.convert_checked.loc16_55: init %u32 = call %Convert.specific_fn.loc16_55(%int_11259375) [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %.loc16_55.1: %u32 = value_of_initializer %int.convert_checked.loc16_55 [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %.loc16_55.2: %u32 = converted %int_11259375, %.loc16_55.1 [template = constants.%int_11259375.1]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_11259375.1) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc16_34.4: ref %Expect.6 = temporary %.loc16_34.3, %Test.call.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %u32, %b.param_patt: %i32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %u32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc20_26.1: %u32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc20_26.2: %u32 = converted %int.right_shift, %.loc20_26.1
// CHECK:STDOUT:   return %.loc20_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_1.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_5.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_5.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_5.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_5.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_5.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_5.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_2147483647.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_11259375.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_11259375.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_11259375.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_11259375.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_11259375.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_11259375.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
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
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_5) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_5) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_-1) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_-1) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2, %Op.2 [template]
// CHECK:STDOUT:   %int_-2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_10, %Op.2 [template]
// CHECK:STDOUT:   %int_-10: Core.IntLiteral = int_value -10 [template]
// CHECK:STDOUT:   %int_-3: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_-3) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_-3) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_3, %Op.2 [template]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_2147483647) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_2147483647) [template]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %int_11259375: Core.IntLiteral = int_value 11259375 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_11259375) [template]
// CHECK:STDOUT:   %Test.specific_fn.7: <specific function> = specific_function %Test, @Test(%int_11259375) [template]
// CHECK:STDOUT:   %int_305419896: Core.IntLiteral = int_value 305419896 [template]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %Op.bound.5: <bound method> = bound_method %int_305419896, %Op.2 [template]
// CHECK:STDOUT:   %int_-305419896: Core.IntLiteral = int_value -305419896 [template]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %Op.bound.6: <bound method> = bound_method %int_9223372036854775807, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775807: Core.IntLiteral = int_value -9223372036854775807 [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Op.bound.7: <bound method> = bound_method %int_9223372036854775808, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.8: %Expect.8 = struct_value () [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_18: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_22: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_34: init type = call %IntLiteral.ref.loc4_22() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int_literal.make_type.loc4_34 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int_literal.make_type.loc4_34, %.loc4_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_40: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_44: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_56: init type = call %IntLiteral.ref.loc4_44() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.1: type = value_of_initializer %int_literal.make_type.loc4_56 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.2: type = converted %int_literal.make_type.loc4_56, %.loc4_56.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_62: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_66: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_78: init type = call %IntLiteral.ref.loc4_66() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_78.1: type = value_of_initializer %int_literal.make_type.loc4_78 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_78.2: type = converted %int_literal.make_type.loc4_78, %.loc4_78.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int_literal.make_type, %.loc6_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc7_29.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc7_29.2: type = converted %int_literal.make_type, %.loc7_29.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %Expect.loc7_43.1: type = class_type @Expect, @Expect(constants.%N) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
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
// CHECK:STDOUT: fn @RightShift(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.right_shift";
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
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_0.loc10_19: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_31.loc10: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %int.right_shift.loc10: init Core.IntLiteral = call %RightShift.ref.loc10(%int_0.loc10_19, %int_31.loc10) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_25.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc10 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_25.2: Core.IntLiteral = converted %int.right_shift.loc10, %.loc10_25.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_25.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_25.4: ref %Expect.2 = temporary %.loc10_25.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc11: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_31.loc11: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %int.right_shift.loc11: init Core.IntLiteral = call %RightShift.ref.loc11(%int_1.loc11, %int_31.loc11) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_25.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc11 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_25.2: Core.IntLiteral = converted %int.right_shift.loc11, %.loc11_25.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_25.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_25.4: ref %Expect.2 = temporary %.loc11_25.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc12: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc12_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.right_shift.loc12: init Core.IntLiteral = call %RightShift.ref.loc12(%int_1.loc12_19, %int_0.loc12) [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_24.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc12 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_24.2: Core.IntLiteral = converted %int.right_shift.loc12, %.loc12_24.1 [template = constants.%int_1]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_24.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc12_36: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_24.4: ref %Expect.3 = temporary %.loc12_24.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.right_shift.loc13: init Core.IntLiteral = call %RightShift.ref.loc13(%int_1.loc13, %int_2.loc13) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_24.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc13 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_24.2: Core.IntLiteral = converted %int.right_shift.loc13, %.loc13_24.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_24.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_24.4: ref %Expect.2 = temporary %.loc13_24.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template = constants.%int_22]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.right_shift.loc14: init Core.IntLiteral = call %RightShift.ref.loc14(%int_22, %int_2.loc14) [template = constants.%int_5]
// CHECK:STDOUT:   %.loc14_25.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc14 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc14_25.2: Core.IntLiteral = converted %int.right_shift.loc14, %.loc14_25.1 [template = constants.%int_5]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_5) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_25.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_25.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_5) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_25.4: ref %Expect.4 = temporary %.loc14_25.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc15: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc15_20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_19: <bound method> = bound_method %int_1.loc15_20, %impl.elem0.loc15_19 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_19: init Core.IntLiteral = call %Op.bound.loc15_19(%int_1.loc15_20) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc15_23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc15_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_19 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_19.2: Core.IntLiteral = converted %int.snegate.loc15_19, %.loc15_19.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.right_shift.loc15: init Core.IntLiteral = call %RightShift.ref.loc15(%.loc15_19.2, %int_1.loc15_23) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_25.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_25.2: Core.IntLiteral = converted %int.right_shift.loc15, %.loc15_25.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_25.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_25.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc15_38: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15_37: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15_37: <bound method> = bound_method %int_1.loc15_38, %impl.elem0.loc15_37 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_37: init Core.IntLiteral = call %Op.bound.loc15_37(%int_1.loc15_38) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_39.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_37 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_39.2: Core.IntLiteral = converted %int.snegate.loc15_37, %.loc15_39.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_25.4: ref %Expect.5 = temporary %.loc15_25.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc16: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc16_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16_19: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_19 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc16_19: init Core.IntLiteral = call %Op.bound.loc16_19(%int_2.loc16) [template = constants.%int_-2]
// CHECK:STDOUT:   %int_1.loc16_23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc16_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_19 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc16_19.2: Core.IntLiteral = converted %int.snegate.loc16_19, %.loc16_19.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %int.right_shift.loc16: init Core.IntLiteral = call %RightShift.ref.loc16(%.loc16_19.2, %int_1.loc16_23) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_25.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc16 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_25.2: Core.IntLiteral = converted %int.right_shift.loc16, %.loc16_25.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc16_25.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.5 = call %Test.specific_fn.loc16() to %.loc16_25.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc16_38: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc16_37: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16_37: <bound method> = bound_method %int_1.loc16_38, %impl.elem0.loc16_37 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc16_37: init Core.IntLiteral = call %Op.bound.loc16_37(%int_1.loc16_38) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_39.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_37 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_39.2: Core.IntLiteral = converted %int.snegate.loc16_37, %.loc16_39.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc16_25.4: ref %Expect.5 = temporary %.loc16_25.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc17: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc17_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc17_19: <bound method> = bound_method %int_10, %impl.elem0.loc17_19 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc17_19: init Core.IntLiteral = call %Op.bound.loc17_19(%int_10) [template = constants.%int_-10]
// CHECK:STDOUT:   %int_2.loc17: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc17_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_19 [template = constants.%int_-10]
// CHECK:STDOUT:   %.loc17_19.2: Core.IntLiteral = converted %int.snegate.loc17_19, %.loc17_19.1 [template = constants.%int_-10]
// CHECK:STDOUT:   %int.right_shift.loc17: init Core.IntLiteral = call %RightShift.ref.loc17(%.loc17_19.2, %int_2.loc17) [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc17_26.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc17 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc17_26.2: Core.IntLiteral = converted %int.right_shift.loc17, %.loc17_26.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_-3) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc17_26.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.6 = call %Test.specific_fn.loc17() to %.loc17_26.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc17_38: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc17_38: <bound method> = bound_method %int_3, %impl.elem0.loc17_38 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc17_38: init Core.IntLiteral = call %Op.bound.loc17_38(%int_3) [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc17_40.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_38 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc17_40.2: Core.IntLiteral = converted %int.snegate.loc17_38, %.loc17_40.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_-3) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc17_26.4: ref %Expect.6 = temporary %.loc17_26.3, %Test.call.loc17
// CHECK:STDOUT:   %Test.ref.loc18: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc18: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template = constants.%int_4294967295]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.right_shift.loc18: init Core.IntLiteral = call %RightShift.ref.loc18(%int_4294967295, %int_1.loc18) [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc18_34.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc18 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc18_34.2: Core.IntLiteral = converted %int.right_shift.loc18, %.loc18_34.1 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Test.specific_fn.loc18: <specific function> = specific_function %Test.ref.loc18, @Test(constants.%int_2147483647) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc18_34.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc18: init %Expect.7 = call %Test.specific_fn.loc18() to %.loc18_34.3
// CHECK:STDOUT:   %Expect.ref.loc18: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Expect.loc18: type = class_type @Expect, @Expect(constants.%int_2147483647) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc18_34.4: ref %Expect.7 = temporary %.loc18_34.3, %Test.call.loc18
// CHECK:STDOUT:   %Test.ref.loc19: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc19: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template = constants.%int_2882400001]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %int.right_shift.loc19: init Core.IntLiteral = call %RightShift.ref.loc19(%int_2882400001, %int_8) [template = constants.%int_11259375]
// CHECK:STDOUT:   %.loc19_34.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc19 [template = constants.%int_11259375]
// CHECK:STDOUT:   %.loc19_34.2: Core.IntLiteral = converted %int.right_shift.loc19, %.loc19_34.1 [template = constants.%int_11259375]
// CHECK:STDOUT:   %Test.specific_fn.loc19: <specific function> = specific_function %Test.ref.loc19, @Test(constants.%int_11259375) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc19_34.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc19: init %Expect.8 = call %Test.specific_fn.loc19() to %.loc19_34.3
// CHECK:STDOUT:   %Expect.ref.loc19: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_11259375: Core.IntLiteral = int_value 11259375 [template = constants.%int_11259375]
// CHECK:STDOUT:   %Expect.loc19: type = class_type @Expect, @Expect(constants.%int_11259375) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc19_34.4: ref %Expect.8 = temporary %.loc19_34.3, %Test.call.loc19
// CHECK:STDOUT:   %Test.ref.loc21: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc21: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_305419896.loc21: Core.IntLiteral = int_value 305419896 [template = constants.%int_305419896]
// CHECK:STDOUT:   %int_1000000000.loc21: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.right_shift.loc21: init Core.IntLiteral = call %RightShift.ref.loc21(%int_305419896.loc21, %int_1000000000.loc21) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc21_46.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc21 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc21_46.2: Core.IntLiteral = converted %int.right_shift.loc21, %.loc21_46.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc21: <specific function> = specific_function %Test.ref.loc21, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc21_46.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc21: init %Expect.2 = call %Test.specific_fn.loc21() to %.loc21_46.3
// CHECK:STDOUT:   %Expect.ref.loc21: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc21: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc21_46.4: ref %Expect.2 = temporary %.loc21_46.3, %Test.call.loc21
// CHECK:STDOUT:   %Test.ref.loc22: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc22: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_305419896.loc22: Core.IntLiteral = int_value 305419896 [template = constants.%int_305419896]
// CHECK:STDOUT:   %impl.elem0.loc22_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc22_19: <bound method> = bound_method %int_305419896.loc22, %impl.elem0.loc22_19 [template = constants.%Op.bound.5]
// CHECK:STDOUT:   %int.snegate.loc22_19: init Core.IntLiteral = call %Op.bound.loc22_19(%int_305419896.loc22) [template = constants.%int_-305419896]
// CHECK:STDOUT:   %int_1000000000.loc22: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %.loc22_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc22_19 [template = constants.%int_-305419896]
// CHECK:STDOUT:   %.loc22_19.2: Core.IntLiteral = converted %int.snegate.loc22_19, %.loc22_19.1 [template = constants.%int_-305419896]
// CHECK:STDOUT:   %int.right_shift.loc22: init Core.IntLiteral = call %RightShift.ref.loc22(%.loc22_19.2, %int_1000000000.loc22) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_47.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc22 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_47.2: Core.IntLiteral = converted %int.right_shift.loc22, %.loc22_47.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc22: <specific function> = specific_function %Test.ref.loc22, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc22_47.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc22: init %Expect.5 = call %Test.specific_fn.loc22() to %.loc22_47.3
// CHECK:STDOUT:   %Expect.ref.loc22: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc22_59: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc22_59: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_59 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc22_59: init Core.IntLiteral = call %Op.bound.loc22_59(%int_1.loc22) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_61.1: Core.IntLiteral = value_of_initializer %int.snegate.loc22_59 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc22_61.2: Core.IntLiteral = converted %int.snegate.loc22_59, %.loc22_61.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc22: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc22_47.4: ref %Expect.5 = temporary %.loc22_47.3, %Test.call.loc22
// CHECK:STDOUT:   %Test.ref.loc23: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc23: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template = constants.%int_18446744073709551615]
// CHECK:STDOUT:   %int_1000000000.loc23: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.right_shift.loc23: init Core.IntLiteral = call %RightShift.ref.loc23(%int_18446744073709551615, %int_1000000000.loc23) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc23_56.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc23 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc23_56.2: Core.IntLiteral = converted %int.right_shift.loc23, %.loc23_56.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc23: <specific function> = specific_function %Test.ref.loc23, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc23_56.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc23: init %Expect.2 = call %Test.specific_fn.loc23() to %.loc23_56.3
// CHECK:STDOUT:   %Expect.ref.loc23: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc23: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc23_56.4: ref %Expect.2 = temporary %.loc23_56.3, %Test.call.loc23
// CHECK:STDOUT:   %Test.ref.loc24: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc24: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_9223372036854775807.loc24: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int_1000000000.loc24: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.right_shift.loc24: init Core.IntLiteral = call %RightShift.ref.loc24(%int_9223372036854775807.loc24, %int_1000000000.loc24) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc24_56.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc24 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc24_56.2: Core.IntLiteral = converted %int.right_shift.loc24, %.loc24_56.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc24: <specific function> = specific_function %Test.ref.loc24, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc24_56.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc24: init %Expect.2 = call %Test.specific_fn.loc24() to %.loc24_56.3
// CHECK:STDOUT:   %Expect.ref.loc24: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc24: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc24: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc24_56.4: ref %Expect.2 = temporary %.loc24_56.3, %Test.call.loc24
// CHECK:STDOUT:   %Test.ref.loc25: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc25: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_9223372036854775807.loc25: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %impl.elem0.loc25_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc25_19: <bound method> = bound_method %int_9223372036854775807.loc25, %impl.elem0.loc25_19 [template = constants.%Op.bound.6]
// CHECK:STDOUT:   %int.snegate.loc25_19: init Core.IntLiteral = call %Op.bound.loc25_19(%int_9223372036854775807.loc25) [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %int_1000000000.loc25: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %.loc25_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc25_19 [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %.loc25_19.2: Core.IntLiteral = converted %int.snegate.loc25_19, %.loc25_19.1 [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %int.right_shift.loc25: init Core.IntLiteral = call %RightShift.ref.loc25(%.loc25_19.2, %int_1000000000.loc25) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc25_57.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc25 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc25_57.2: Core.IntLiteral = converted %int.right_shift.loc25, %.loc25_57.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc25: <specific function> = specific_function %Test.ref.loc25, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc25_57.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc25: init %Expect.5 = call %Test.specific_fn.loc25() to %.loc25_57.3
// CHECK:STDOUT:   %Expect.ref.loc25: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc25_69: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc25_69: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25_69 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc25_69: init Core.IntLiteral = call %Op.bound.loc25_69(%int_1.loc25) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc25_71.1: Core.IntLiteral = value_of_initializer %int.snegate.loc25_69 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc25_71.2: Core.IntLiteral = converted %int.snegate.loc25_69, %.loc25_71.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc25: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc25_57.4: ref %Expect.5 = temporary %.loc25_57.3, %Test.call.loc25
// CHECK:STDOUT:   %Test.ref.loc26: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %RightShift.ref.loc26: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc26_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc26_19: <bound method> = bound_method %int_9223372036854775808, %impl.elem0.loc26_19 [template = constants.%Op.bound.7]
// CHECK:STDOUT:   %int.snegate.loc26_19: init Core.IntLiteral = call %Op.bound.loc26_19(%int_9223372036854775808) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1000000000.loc26: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %.loc26_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc26_19 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc26_19.2: Core.IntLiteral = converted %int.snegate.loc26_19, %.loc26_19.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.right_shift.loc26: init Core.IntLiteral = call %RightShift.ref.loc26(%.loc26_19.2, %int_1000000000.loc26) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_57.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc26 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_57.2: Core.IntLiteral = converted %int.right_shift.loc26, %.loc26_57.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc26: <specific function> = specific_function %Test.ref.loc26, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc26_57.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc26: init %Expect.5 = call %Test.specific_fn.loc26() to %.loc26_57.3
// CHECK:STDOUT:   %Expect.ref.loc26: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc26_69: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc26_69: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_69 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc26_69: init Core.IntLiteral = call %Op.bound.loc26_69(%int_1.loc26) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_71.1: Core.IntLiteral = value_of_initializer %int.snegate.loc26_69 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_71.2: Core.IntLiteral = converted %int.snegate.loc26_69, %.loc26_71.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc26: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc26_57.4: ref %Expect.5 = temporary %.loc26_57.3, %Test.call.loc26
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
// CHECK:STDOUT: specific @Test(constants.%int_1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_5) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_5
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_5
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_5) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_5
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-3) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-3
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-3
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-3) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-3
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_11259375) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_11259375
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_11259375
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_11259375) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_11259375
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_11259375
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32.1) [template]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %RightShiftLit.type: type = fn_type @RightShiftLit [template]
// CHECK:STDOUT:   %RightShiftLit: %RightShiftLit.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32.1) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %int_0.1: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %int_33.1: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_33.2: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %int_0.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_0.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IntLiteral = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     .Negate = %import_ref.230
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .RightShiftLit = %RightShiftLit.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:     .negative_zero = @__global_init.%negative_zero
// CHECK:STDOUT:     .negative_lit = @__global_init.%negative_lit
// CHECK:STDOUT:     .negative_lit_zero = @__global_init.%negative_lit_zero
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_26: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_26: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_34: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_34: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RightShiftLit.decl: %RightShiftLit.type = fn_decl @RightShiftLit [template = constants.%RightShiftLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc5_21: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc5_25: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc5_37: init type = call %IntLiteral.ref.loc5_25() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_37.1: type = value_of_initializer %int_literal.make_type.loc5_37 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %int_literal.make_type.loc5_37, %.loc5_37.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %Core.ref.loc5_51: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc5_55: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc5_67: init type = call %IntLiteral.ref.loc5_55() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_67.1: type = value_of_initializer %int_literal.make_type.loc5_67 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_67.2: type = converted %int_literal.make_type.loc5_67, %.loc5_67.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc25: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc30: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc30: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %Core.ref.loc35: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %IntLiteral.ref.loc35: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:   %int_literal.make_type.loc35: init type = call %IntLiteral.ref.loc35() [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc35_35.1: type = value_of_initializer %int_literal.make_type.loc35 [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc35_35.2: type = converted %int_literal.make_type.loc35, %.loc35_35.1 [template = Core.IntLiteral]
// CHECK:STDOUT:   %Core.ref.loc39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %IntLiteral.ref.loc39: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:   %int_literal.make_type.loc39: init type = call %IntLiteral.ref.loc39() [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc39_40.1: type = value_of_initializer %int_literal.make_type.loc39 [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc39_40.2: type = converted %int_literal.make_type.loc39, %.loc39_40.1 [template = Core.IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShiftLit(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> Core.IntLiteral = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref.loc8: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc8_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_30: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_30: <specific function> = specific_function %Convert.bound.loc8_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_30: init %i32 = call %Convert.specific_fn.loc8_30(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_30.1: %i32 = value_of_initializer %int.convert_checked.loc8_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_30.2: %i32 = converted %int_1.loc8, %.loc8_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc8_33: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_33: <bound method> = bound_method %int_31, %impl.elem0.loc8_33 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_33: <specific function> = specific_function %Convert.bound.loc8_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_33: init %i32 = call %Convert.specific_fn.loc8_33(%int_31) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_33.1: %i32 = value_of_initializer %int.convert_checked.loc8_33 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_33.2: %i32 = converted %int_31, %.loc8_33.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc8: init %i32 = call %RightShift.ref.loc8(%.loc8_30.2, %.loc8_33.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc8_36.1: %i32 = value_of_initializer %int.right_shift.loc8 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc8_36.2: %i32 = converted %int.right_shift.loc8, %.loc8_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_36.2
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc13_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_30: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_30: <specific function> = specific_function %Convert.bound.loc13_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_30: init %i32 = call %Convert.specific_fn.loc13_30(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_30.1: %i32 = value_of_initializer %int.convert_checked.loc13_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_30.2: %i32 = converted %int_1.loc13, %.loc13_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_33: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_33: <bound method> = bound_method %int_32, %impl.elem0.loc13_33 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_33: <specific function> = specific_function %Convert.bound.loc13_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_33: init %i32 = call %Convert.specific_fn.loc13_33(%int_32) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_33.1: %i32 = value_of_initializer %int.convert_checked.loc13_33 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = converted %int_32, %.loc13_33.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.right_shift.loc13: init %i32 = call %RightShift.ref.loc13(%.loc13_30.2, %.loc13_33.2) [template = <error>]
// CHECK:STDOUT:   %.loc13_36.1: %i32 = value_of_initializer %int.right_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_36.2: %i32 = converted %int.right_shift.loc13, %.loc13_36.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_36.2
// CHECK:STDOUT:   %RightShift.ref.loc18: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33.1]
// CHECK:STDOUT:   %impl.elem0.loc18_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_30: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_30: <specific function> = specific_function %Convert.bound.loc18_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_30: init %i32 = call %Convert.specific_fn.loc18_30(%int_1.loc18) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_30.1: %i32 = value_of_initializer %int.convert_checked.loc18_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_30.2: %i32 = converted %int_1.loc18, %.loc18_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_33: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_33: <bound method> = bound_method %int_33, %impl.elem0.loc18_33 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_33: <specific function> = specific_function %Convert.bound.loc18_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc18_33: init %i32 = call %Convert.specific_fn.loc18_33(%int_33) [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_33.1: %i32 = value_of_initializer %int.convert_checked.loc18_33 [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_33.2: %i32 = converted %int_33, %.loc18_33.1 [template = constants.%int_33.2]
// CHECK:STDOUT:   %int.right_shift.loc18: init %i32 = call %RightShift.ref.loc18(%.loc18_30.2, %.loc18_33.2) [template = <error>]
// CHECK:STDOUT:   %.loc18_36.1: %i32 = value_of_initializer %int.right_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_36.2: %i32 = converted %int.right_shift.loc18, %.loc18_36.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_36.2
// CHECK:STDOUT:   %RightShift.ref.loc25: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc25_32: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc25_36: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc25_35.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc25: <bound method> = bound_method %int_1.loc25_36, %impl.elem0.loc25_35.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc25: init Core.IntLiteral = call %Op.bound.loc25(%int_1.loc25_36) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc25_32: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc25_32: <bound method> = bound_method %int_1.loc25_32, %impl.elem0.loc25_32 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc25_32: <specific function> = specific_function %Convert.bound.loc25_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc25_32: init %i32 = call %Convert.specific_fn.loc25_32(%int_1.loc25_32) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc25_32.1: %i32 = value_of_initializer %int.convert_checked.loc25_32 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc25_32.2: %i32 = converted %int_1.loc25_32, %.loc25_32.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc25_35.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc25_35: <bound method> = bound_method %int.snegate.loc25, %impl.elem0.loc25_35.2 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc25_35: <specific function> = specific_function %Convert.bound.loc25_35, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %.loc25_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc25 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc25_35.2: Core.IntLiteral = converted %int.snegate.loc25, %.loc25_35.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc25_35: init %i32 = call %Convert.specific_fn.loc25_35(%.loc25_35.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc25_35.3: %i32 = value_of_initializer %int.convert_checked.loc25_35 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc25_35.4: %i32 = converted %int.snegate.loc25, %.loc25_35.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.right_shift.loc25: init %i32 = call %RightShift.ref.loc25(%.loc25_32.2, %.loc25_35.4) [template = <error>]
// CHECK:STDOUT:   %.loc25_38.1: %i32 = value_of_initializer %int.right_shift.loc25 [template = <error>]
// CHECK:STDOUT:   %.loc25_38.2: %i32 = converted %int.right_shift.loc25, %.loc25_38.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc25_38.2
// CHECK:STDOUT:   %RightShift.ref.loc30: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_0.loc30: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc30_40.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc30: <bound method> = bound_method %int_1.loc30, %impl.elem0.loc30_40.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc30: init Core.IntLiteral = call %Op.bound.loc30(%int_1.loc30) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc30_37: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc30_37: <bound method> = bound_method %int_0.loc30, %impl.elem0.loc30_37 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_37: <specific function> = specific_function %Convert.bound.loc30_37, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc30_37: init %i32 = call %Convert.specific_fn.loc30_37(%int_0.loc30) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc30_37.1: %i32 = value_of_initializer %int.convert_checked.loc30_37 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc30_37.2: %i32 = converted %int_0.loc30, %.loc30_37.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc30_40.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc30_40: <bound method> = bound_method %int.snegate.loc30, %impl.elem0.loc30_40.2 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_40: <specific function> = specific_function %Convert.bound.loc30_40, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %.loc30_40.1: Core.IntLiteral = value_of_initializer %int.snegate.loc30 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc30_40.2: Core.IntLiteral = converted %int.snegate.loc30, %.loc30_40.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc30_40: init %i32 = call %Convert.specific_fn.loc30_40(%.loc30_40.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc30_40.3: %i32 = value_of_initializer %int.convert_checked.loc30_40 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc30_40.4: %i32 = converted %int.snegate.loc30, %.loc30_40.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.right_shift.loc30: init %i32 = call %RightShift.ref.loc30(%.loc30_37.2, %.loc30_40.4) [template = <error>]
// CHECK:STDOUT:   %.loc30_43.1: %i32 = value_of_initializer %int.right_shift.loc30 [template = <error>]
// CHECK:STDOUT:   %.loc30_43.2: %i32 = converted %int.right_shift.loc30, %.loc30_43.1 [template = <error>]
// CHECK:STDOUT:   %negative_zero: %i32 = bind_name negative_zero, %.loc30_43.2
// CHECK:STDOUT:   %RightShiftLit.ref.loc35: %RightShiftLit.type = name_ref RightShiftLit, file.%RightShiftLit.decl [template = constants.%RightShiftLit]
// CHECK:STDOUT:   %int_1.loc35_53: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc35_57: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc35_56.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc35: <bound method> = bound_method %int_1.loc35_57, %impl.elem0.loc35_56.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc35: init Core.IntLiteral = call %Op.bound.loc35(%int_1.loc35_57) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc35_56.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc35: <bound method> = bound_method %int.snegate.loc35, %impl.elem0.loc35_56.2 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc35: <specific function> = specific_function %Convert.bound.loc35, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %.loc35_56.1: Core.IntLiteral = value_of_initializer %int.snegate.loc35 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc35_56.2: Core.IntLiteral = converted %int.snegate.loc35, %.loc35_56.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc35: init %i32 = call %Convert.specific_fn.loc35(%.loc35_56.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc35_56.3: %i32 = value_of_initializer %int.convert_checked.loc35 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc35_56.4: %i32 = converted %int.snegate.loc35, %.loc35_56.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.right_shift.loc35: init Core.IntLiteral = call %RightShiftLit.ref.loc35(%int_1.loc35_53, %.loc35_56.4) [template = <error>]
// CHECK:STDOUT:   %.loc35_59.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc35 [template = <error>]
// CHECK:STDOUT:   %.loc35_59.2: Core.IntLiteral = converted %int.right_shift.loc35, %.loc35_59.1 [template = <error>]
// CHECK:STDOUT:   %negative_lit: Core.IntLiteral = bind_name negative_lit, %.loc35_59.2
// CHECK:STDOUT:   %RightShiftLit.ref.loc39: %RightShiftLit.type = name_ref RightShiftLit, file.%RightShiftLit.decl [template = constants.%RightShiftLit]
// CHECK:STDOUT:   %int_0.loc39: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_1.loc39: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc39_61.1: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc39: <bound method> = bound_method %int_1.loc39, %impl.elem0.loc39_61.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc39: init Core.IntLiteral = call %Op.bound.loc39(%int_1.loc39) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc39_61.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc39: <bound method> = bound_method %int.snegate.loc39, %impl.elem0.loc39_61.2 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc39: <specific function> = specific_function %Convert.bound.loc39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %.loc39_61.1: Core.IntLiteral = value_of_initializer %int.snegate.loc39 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc39_61.2: Core.IntLiteral = converted %int.snegate.loc39, %.loc39_61.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc39: init %i32 = call %Convert.specific_fn.loc39(%.loc39_61.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc39_61.3: %i32 = value_of_initializer %int.convert_checked.loc39 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc39_61.4: %i32 = converted %int.snegate.loc39, %.loc39_61.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.right_shift.loc39: init Core.IntLiteral = call %RightShiftLit.ref.loc39(%int_0.loc39, %.loc39_61.4) [template = <error>]
// CHECK:STDOUT:   %.loc39_64.1: Core.IntLiteral = value_of_initializer %int.right_shift.loc39 [template = <error>]
// CHECK:STDOUT:   %.loc39_64.2: Core.IntLiteral = converted %int.right_shift.loc39, %.loc39_64.1 [template = <error>]
// CHECK:STDOUT:   %negative_lit_zero: Core.IntLiteral = bind_name negative_lit_zero, %.loc39_64.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
