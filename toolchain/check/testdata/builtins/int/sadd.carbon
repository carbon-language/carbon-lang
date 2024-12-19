// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sadd.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
}

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);

  // Test some cases that might -- but shouldn't -- overflow.
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
  Test(Add(0x7FFF_FFFF, 1)) as Expect(0x8000_0000);
  Test(Add(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0x8000_0000_0000_0000);
  Test(Add(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_0000_0000_0000_0000);
  Test(Add(-0x8000_0000_0000_0000, -1)) as Expect(-0x8000_0000_0000_0001);
  Test(Add(-0x8000_0000_0000_0000, -0x8000_0000_0000_0000)) as Expect(-0x1_0000_0000_0000_0000);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: [i32; TooFew(1)];
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew(1)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1, 2)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1, 2)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2, 3)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

let a: i32 = Add(0x7FFFFFFF, 0);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in calculation `2147483647 + 1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Add(0x7FFFFFFF, 1);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
let b: i32 = Add(0x7FFFFFFF, 1);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Add.type.1: type = fn_type @Add.1 [template]
// CHECK:STDOUT:   %Add: %Add.type.1 = struct_value () [template]
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
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_3.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_3.1) [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483646.1: Core.IntLiteral = int_value 2147483646 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_2147483646.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483646.2: %i32 = int_value 2147483646 [template]
// CHECK:STDOUT:   %int_2147483647.1: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_2147483647.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_2147483647.1) [template]
// CHECK:STDOUT:   %int_2147483647.2: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_2147483647.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type.1 = fn_decl @Add.1 [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_14.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc7_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc7_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
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
// CHECK:STDOUT:     %int_32.loc15_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
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
// CHECK:STDOUT: fn @Add.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
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
// CHECK:STDOUT:   %Add.ref.loc10: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_0.loc10_12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc10_15: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_12: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_12: <bound method> = bound_method %int_0.loc10_12, %impl.elem0.loc10_12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_12: <specific function> = specific_function %Convert.bound.loc10_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_12: init %i32 = call %Convert.specific_fn.loc10_12(%int_0.loc10_12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_12.1: %i32 = value_of_initializer %int.convert_checked.loc10_12 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_12.2: %i32 = converted %int_0.loc10_12, %.loc10_12.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_15: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_15: <bound method> = bound_method %int_0.loc10_15, %impl.elem0.loc10_15 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_15: <specific function> = specific_function %Convert.bound.loc10_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_15: init %i32 = call %Convert.specific_fn.loc10_15(%int_0.loc10_15) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_15.1: %i32 = value_of_initializer %int.convert_checked.loc10_15 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = converted %int_0.loc10_15, %.loc10_15.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.sadd.loc10: init %i32 = call %Add.ref.loc10(%.loc10_12.2, %.loc10_15.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_17.1: %i32 = value_of_initializer %int.sadd.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_17.2: %i32 = converted %int.sadd.loc10, %.loc10_17.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_29: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_30: <bound method> = bound_method %int_0.loc10_29, %impl.elem0.loc10_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_30: <specific function> = specific_function %Convert.bound.loc10_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_30: init %i32 = call %Convert.specific_fn.loc10_30(%int_0.loc10_29) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %int.convert_checked.loc10_30 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_0.loc10_29, %.loc10_30.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc11: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc11_12: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_12: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_12 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_12: <specific function> = specific_function %Convert.bound.loc11_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_12: init %i32 = call %Convert.specific_fn.loc11_12(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_12.1: %i32 = value_of_initializer %int.convert_checked.loc11_12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_12.2: %i32 = converted %int_1.loc11, %.loc11_12.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_15: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_15: <bound method> = bound_method %int_2, %impl.elem0.loc11_15 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_15: <specific function> = specific_function %Convert.bound.loc11_15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_15: init %i32 = call %Convert.specific_fn.loc11_15(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_15.1: %i32 = value_of_initializer %int.convert_checked.loc11_15 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_15.2: %i32 = converted %int_2, %.loc11_15.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.sadd.loc11: init %i32 = call %Add.ref.loc11(%.loc11_12.2, %.loc11_15.2) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_17.1: %i32 = value_of_initializer %int.sadd.loc11 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_17.2: %i32 = converted %int.sadd.loc11, %.loc11_17.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_3.1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_17.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %impl.elem0.loc11_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_30: <bound method> = bound_method %int_3, %impl.elem0.loc11_30 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_30: <specific function> = specific_function %Convert.bound.loc11_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc11_30: init %i32 = call %Convert.specific_fn.loc11_30(%int_3) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_30.1: %i32 = value_of_initializer %int.convert_checked.loc11_30 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_30.2: %i32 = converted %int_3, %.loc11_30.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_3.1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_17.4: ref %Expect.3 = temporary %.loc11_17.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc12: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_2147483646: Core.IntLiteral = int_value 2147483646 [template = constants.%int_2147483646.1]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_12: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_12: <bound method> = bound_method %int_2147483646, %impl.elem0.loc12_12 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_12: <specific function> = specific_function %Convert.bound.loc12_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc12_12: init %i32 = call %Convert.specific_fn.loc12_12(%int_2147483646) [template = constants.%int_2147483646.2]
// CHECK:STDOUT:   %.loc12_12.1: %i32 = value_of_initializer %int.convert_checked.loc12_12 [template = constants.%int_2147483646.2]
// CHECK:STDOUT:   %.loc12_12.2: %i32 = converted %int_2147483646, %.loc12_12.1 [template = constants.%int_2147483646.2]
// CHECK:STDOUT:   %impl.elem0.loc12_25: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_25: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_25 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_25: <specific function> = specific_function %Convert.bound.loc12_25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_25: init %i32 = call %Convert.specific_fn.loc12_25(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_25.1: %i32 = value_of_initializer %int.convert_checked.loc12_25 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_25.2: %i32 = converted %int_1.loc12, %.loc12_25.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.sadd.loc12: init %i32 = call %Add.ref.loc12(%.loc12_12.2, %.loc12_25.2) [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc12_27.1: %i32 = value_of_initializer %int.sadd.loc12 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc12_27.2: %i32 = converted %int.sadd.loc12, %.loc12_27.1 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_2147483647.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc12_27.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.4 = call %Test.specific_fn.loc12() to %.loc12_27.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc12_50: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_50: <bound method> = bound_method %int_2147483647, %impl.elem0.loc12_50 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_50: <specific function> = specific_function %Convert.bound.loc12_50, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc12_50: init %i32 = call %Convert.specific_fn.loc12_50(%int_2147483647) [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc12_50.1: %i32 = value_of_initializer %int.convert_checked.loc12_50 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc12_50.2: %i32 = converted %int_2147483647, %.loc12_50.1 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_2147483647.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc12_27.4: ref %Expect.4 = temporary %.loc12_27.3, %Test.call.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sadd: init %i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_19.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc16_19.2: %i32 = converted %int.sadd, %.loc16_19.1
// CHECK:STDOUT:   return %.loc16_19.2
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
// CHECK:STDOUT: specific @Test(constants.%int_3.1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_3.1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_3.1
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_3.1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_3.1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647.1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647.1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647.1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type.1: type = fn_type @Add.1 [template]
// CHECK:STDOUT:   %Add: %Add.type.1 = struct_value () [template]
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_3) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_3) [template]
// CHECK:STDOUT:   %int_2147483646: Core.IntLiteral = int_value 2147483646 [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_2147483647) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_2147483647) [template]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_2147483648) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_2147483648) [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template]
// CHECK:STDOUT:   %int_18446744073709551616: Core.IntLiteral = int_value 18446744073709551616 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_18446744073709551616) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_18446744073709551616) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_9223372036854775808, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %int_-9223372036854775809: Core.IntLiteral = int_value -9223372036854775809 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_-9223372036854775809) [template]
// CHECK:STDOUT:   %Test.specific_fn.7: <specific function> = specific_function %Test, @Test(%int_-9223372036854775809) [template]
// CHECK:STDOUT:   %int_9223372036854775809: Core.IntLiteral = int_value 9223372036854775809 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_9223372036854775809, %Op.2 [template]
// CHECK:STDOUT:   %int_-18446744073709551616: Core.IntLiteral = int_value -18446744073709551616 [template]
// CHECK:STDOUT:   %Expect.9: type = class_type @Expect, @Expect(%int_-18446744073709551616) [template]
// CHECK:STDOUT:   %Test.specific_fn.8: <specific function> = specific_function %Test, @Test(%int_-18446744073709551616) [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_18446744073709551616, %Op.2 [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type.1 = fn_decl @Add.1 [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_11: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_15: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_27: init type = call %IntLiteral.ref.loc4_15() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int_literal.make_type.loc4_27 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int_literal.make_type.loc4_27, %.loc4_27.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_33: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_37: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_49: init type = call %IntLiteral.ref.loc4_37() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_49.1: type = value_of_initializer %int_literal.make_type.loc4_49 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_49.2: type = converted %int_literal.make_type.loc4_49, %.loc4_49.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_55: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_59: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_71: init type = call %IntLiteral.ref.loc4_59() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.1: type = value_of_initializer %int_literal.make_type.loc4_71 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.2: type = converted %int_literal.make_type.loc4_71, %.loc4_71.1 [template = Core.IntLiteral]
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
// CHECK:STDOUT: fn @Add.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.sadd";
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
// CHECK:STDOUT:   %Add.ref.loc10: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_0.loc10_12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc10_15: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.sadd.loc10: init Core.IntLiteral = call %Add.ref.loc10(%int_0.loc10_12, %int_0.loc10_15) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.1: Core.IntLiteral = value_of_initializer %int.sadd.loc10 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.2: Core.IntLiteral = converted %int.sadd.loc10, %.loc10_17.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc11: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.sadd.loc11: init Core.IntLiteral = call %Add.ref.loc11(%int_1.loc11, %int_2) [template = constants.%int_3]
// CHECK:STDOUT:   %.loc11_17.1: Core.IntLiteral = value_of_initializer %int.sadd.loc11 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc11_17.2: Core.IntLiteral = converted %int.sadd.loc11, %.loc11_17.1 [template = constants.%int_3]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_3) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_17.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_3) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_17.4: ref %Expect.3 = temporary %.loc11_17.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc14: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_2147483646: Core.IntLiteral = int_value 2147483646 [template = constants.%int_2147483646]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.sadd.loc14: init Core.IntLiteral = call %Add.ref.loc14(%int_2147483646, %int_1.loc14) [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc14_27.1: Core.IntLiteral = value_of_initializer %int.sadd.loc14 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %.loc14_27.2: Core.IntLiteral = converted %int.sadd.loc14, %.loc14_27.1 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_2147483647) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_27.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_27.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc14: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_2147483647) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_27.4: ref %Expect.4 = temporary %.loc14_27.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc15: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_2147483647.loc15: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.sadd.loc15: init Core.IntLiteral = call %Add.ref.loc15(%int_2147483647.loc15, %int_1.loc15) [template = constants.%int_2147483648]
// CHECK:STDOUT:   %.loc15_27.1: Core.IntLiteral = value_of_initializer %int.sadd.loc15 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %.loc15_27.2: Core.IntLiteral = converted %int.sadd.loc15, %.loc15_27.1 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_2147483648) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_27.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_27.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_2147483648) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_27.4: ref %Expect.5 = temporary %.loc15_27.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc16: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.sadd.loc16: init Core.IntLiteral = call %Add.ref.loc16(%int_9223372036854775807, %int_1.loc16) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc16_37.1: Core.IntLiteral = value_of_initializer %int.sadd.loc16 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc16_37.2: Core.IntLiteral = converted %int.sadd.loc16, %.loc16_37.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc16_37.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.6 = call %Test.specific_fn.loc16() to %.loc16_37.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc16: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc16_37.4: ref %Expect.6 = temporary %.loc16_37.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc17: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template = constants.%int_18446744073709551615]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.sadd.loc17: init Core.IntLiteral = call %Add.ref.loc17(%int_18446744073709551615, %int_1.loc17) [template = constants.%int_18446744073709551616]
// CHECK:STDOUT:   %.loc17_37.1: Core.IntLiteral = value_of_initializer %int.sadd.loc17 [template = constants.%int_18446744073709551616]
// CHECK:STDOUT:   %.loc17_37.2: Core.IntLiteral = converted %int.sadd.loc17, %.loc17_37.1 [template = constants.%int_18446744073709551616]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_18446744073709551616) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc17_37.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.7 = call %Test.specific_fn.loc17() to %.loc17_37.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_18446744073709551616.loc17: Core.IntLiteral = int_value 18446744073709551616 [template = constants.%int_18446744073709551616]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_18446744073709551616) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc17_37.4: ref %Expect.7 = temporary %.loc17_37.3, %Test.call.loc17
// CHECK:STDOUT:   %Test.ref.loc18: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc18: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_9223372036854775808.loc18: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc18_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc18_12: <bound method> = bound_method %int_9223372036854775808.loc18, %impl.elem0.loc18_12 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc18_12: init Core.IntLiteral = call %Op.bound.loc18_12(%int_9223372036854775808.loc18) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc18_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc18_36: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_36 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc18_36: init Core.IntLiteral = call %Op.bound.loc18_36(%int_1.loc18) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc18_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc18_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc18_12.2: Core.IntLiteral = converted %int.snegate.loc18_12, %.loc18_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc18_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc18_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc18_36.2: Core.IntLiteral = converted %int.snegate.loc18_36, %.loc18_36.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.sadd.loc18: init Core.IntLiteral = call %Add.ref.loc18(%.loc18_12.2, %.loc18_36.2) [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc18_39.1: Core.IntLiteral = value_of_initializer %int.sadd.loc18 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc18_39.2: Core.IntLiteral = converted %int.sadd.loc18, %.loc18_39.1 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %Test.specific_fn.loc18: <specific function> = specific_function %Test.ref.loc18, @Test(constants.%int_-9223372036854775809) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc18_39.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc18: init %Expect.8 = call %Test.specific_fn.loc18() to %.loc18_39.3
// CHECK:STDOUT:   %Expect.ref.loc18: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775809: Core.IntLiteral = int_value 9223372036854775809 [template = constants.%int_9223372036854775809]
// CHECK:STDOUT:   %impl.elem0.loc18_51: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc18_51: <bound method> = bound_method %int_9223372036854775809, %impl.elem0.loc18_51 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc18_51: init Core.IntLiteral = call %Op.bound.loc18_51(%int_9223372036854775809) [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc18_73.1: Core.IntLiteral = value_of_initializer %int.snegate.loc18_51 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc18_73.2: Core.IntLiteral = converted %int.snegate.loc18_51, %.loc18_73.1 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %Expect.loc18: type = class_type @Expect, @Expect(constants.%int_-9223372036854775809) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc18_39.4: ref %Expect.8 = temporary %.loc18_39.3, %Test.call.loc18
// CHECK:STDOUT:   %Test.ref.loc19: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Add.ref.loc19: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_9223372036854775808.loc19_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc19_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_12: <bound method> = bound_method %int_9223372036854775808.loc19_13, %impl.elem0.loc19_12 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc19_12: init Core.IntLiteral = call %Op.bound.loc19_12(%int_9223372036854775808.loc19_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_9223372036854775808.loc19_37: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc19_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_36: <bound method> = bound_method %int_9223372036854775808.loc19_37, %impl.elem0.loc19_36 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc19_36: init Core.IntLiteral = call %Op.bound.loc19_36(%int_9223372036854775808.loc19_37) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_12.2: Core.IntLiteral = converted %int.snegate.loc19_12, %.loc19_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_36 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc19_36.2: Core.IntLiteral = converted %int.snegate.loc19_36, %.loc19_36.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.sadd.loc19: init Core.IntLiteral = call %Add.ref.loc19(%.loc19_12.2, %.loc19_36.2) [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %.loc19_59.1: Core.IntLiteral = value_of_initializer %int.sadd.loc19 [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %.loc19_59.2: Core.IntLiteral = converted %int.sadd.loc19, %.loc19_59.1 [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %Test.specific_fn.loc19: <specific function> = specific_function %Test.ref.loc19, @Test(constants.%int_-18446744073709551616) [template = constants.%Test.specific_fn.8]
// CHECK:STDOUT:   %.loc19_59.3: ref %Expect.9 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc19: init %Expect.9 = call %Test.specific_fn.loc19() to %.loc19_59.3
// CHECK:STDOUT:   %Expect.ref.loc19: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_18446744073709551616.loc19: Core.IntLiteral = int_value 18446744073709551616 [template = constants.%int_18446744073709551616]
// CHECK:STDOUT:   %impl.elem0.loc19_71: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_71: <bound method> = bound_method %int_18446744073709551616.loc19, %impl.elem0.loc19_71 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc19_71: init Core.IntLiteral = call %Op.bound.loc19_71(%int_18446744073709551616.loc19) [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %.loc19_95.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_71 [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %.loc19_95.2: Core.IntLiteral = converted %int.snegate.loc19_71, %.loc19_95.1 [template = constants.%int_-18446744073709551616]
// CHECK:STDOUT:   %Expect.loc19: type = class_type @Expect, @Expect(constants.%int_-18446744073709551616) [template = constants.%Expect.9]
// CHECK:STDOUT:   %.loc19_59.4: ref %Expect.9 = temporary %.loc19_59.3, %Test.call.loc19
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
// CHECK:STDOUT: specific @Test(constants.%int_3) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_3
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_3) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483648) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483648
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483648
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483648) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483648
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483648
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_18446744073709551616) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_18446744073709551616
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_18446744073709551616
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_18446744073709551616) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_18446744073709551616
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_18446744073709551616
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-9223372036854775809) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-9223372036854775809) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-18446744073709551616) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-18446744073709551616
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-18446744073709551616
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-18446744073709551616) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-18446744073709551616
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-18446744073709551616
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [template]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %BadReturnType.type: type = fn_type @BadReturnType [template]
// CHECK:STDOUT:   %BadReturnType: %BadReturnType.type = struct_value () [template]
// CHECK:STDOUT:   %JustRight.type: type = fn_type @JustRight [template]
// CHECK:STDOUT:   %JustRight: %JustRight.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %RuntimeCallTooFew.type: type = fn_type @RuntimeCallTooFew [template]
// CHECK:STDOUT:   %RuntimeCallTooFew: %RuntimeCallTooFew.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallTooMany.type: type = fn_type @RuntimeCallTooMany [template]
// CHECK:STDOUT:   %RuntimeCallTooMany: %RuntimeCallTooMany.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType.type: type = fn_type @RuntimeCallBadReturnType [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType: %RuntimeCallBadReturnType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:     .BadReturnType = %BadReturnType.decl
// CHECK:STDOUT:     .JustRight = %JustRight.decl
// CHECK:STDOUT:     .too_few = %too_few
// CHECK:STDOUT:     .too_many = %too_many
// CHECK:STDOUT:     .bad_return_type = %bad_return_type
// CHECK:STDOUT:     .bad_call = %bad_call
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew.decl
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany.decl
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [template = constants.%TooFew] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc8_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %i32 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_23: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_31: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_39: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_39: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param runtime_param2
// CHECK:STDOUT:     %c: %i32 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_37.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_37.2: type = converted %bool.make_type, %.loc18_37.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, %TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc25: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc25: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc25: <specific function> = specific_function %Convert.bound.loc25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc25: init %i32 = call %Convert.specific_fn.loc25(%int_1.loc25) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc25_27.1: %i32 = value_of_initializer %int.convert_checked.loc25 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc25_27.2: %i32 = converted %int_1.loc25, %.loc25_27.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %TooFew.call: init %i32 = call %TooFew.ref(%.loc25_27.2)
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %int_32.loc30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc30: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, %TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc30: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3.loc30: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc30_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc30_29: <bound method> = bound_method %int_1.loc30, %impl.elem0.loc30_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_29: <specific function> = specific_function %Convert.bound.loc30_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc30_29: init %i32 = call %Convert.specific_fn.loc30_29(%int_1.loc30) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc30_29.1: %i32 = value_of_initializer %int.convert_checked.loc30_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc30_29.2: %i32 = converted %int_1.loc30, %.loc30_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc30_32: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc30_32: <bound method> = bound_method %int_2.loc30, %impl.elem0.loc30_32 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_32: <specific function> = specific_function %Convert.bound.loc30_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc30_32: init %i32 = call %Convert.specific_fn.loc30_32(%int_2.loc30) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc30_32.1: %i32 = value_of_initializer %int.convert_checked.loc30_32 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc30_32.2: %i32 = converted %int_2.loc30, %.loc30_32.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc30_35: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc30_35: <bound method> = bound_method %int_3.loc30, %impl.elem0.loc30_35 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_35: <specific function> = specific_function %Convert.bound.loc30_35, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc30_35: init %i32 = call %Convert.specific_fn.loc30_35(%int_3.loc30) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc30_35.1: %i32 = value_of_initializer %int.convert_checked.loc30_35 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc30_35.2: %i32 = converted %int_3.loc30, %.loc30_35.1 [template = constants.%int_3.2]
// CHECK:STDOUT:   %TooMany.call: init %i32 = call %TooMany.ref(%.loc30_29.2, %.loc30_32.2, %.loc30_35.2)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %int_32.loc35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, %BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %int_1.loc35: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc35: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc35_42: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc35_42: <bound method> = bound_method %int_1.loc35, %impl.elem0.loc35_42 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc35_42: <specific function> = specific_function %Convert.bound.loc35_42, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc35_42: init %i32 = call %Convert.specific_fn.loc35_42(%int_1.loc35) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_42.1: %i32 = value_of_initializer %int.convert_checked.loc35_42 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_42.2: %i32 = converted %int_1.loc35, %.loc35_42.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc35_45: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc35_45: <bound method> = bound_method %int_2.loc35, %impl.elem0.loc35_45 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc35_45: <specific function> = specific_function %Convert.bound.loc35_45, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc35_45: init %i32 = call %Convert.specific_fn.loc35_45(%int_2.loc35) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc35_45.1: %i32 = value_of_initializer %int.convert_checked.loc35_45 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc35_45.2: %i32 = converted %int_2.loc35, %.loc35_45.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35_42.2, %.loc35_45.2)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %int_32.loc44: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc44: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %JustRight.ref: %JustRight.type = name_ref JustRight, %JustRight.decl [template = constants.%JustRight]
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3.loc44: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %array_type: type = array_type <error>, %i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc46_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc46_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc46_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc46_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %i32 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc50_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc50_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc50_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc50_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc50_42: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc50_42: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc50_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc50_50: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param runtime_param2
// CHECK:STDOUT:     %c: %i32 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc54_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc54_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc54_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc54_40: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc54_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc54_48.2: type = converted %bool.make_type, %.loc54_48.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: %i32, %b.param_patt: %i32, %c.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: %i32, %b.param_patt: %i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init %i32 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc47_19.1: %i32 = value_of_initializer %TooFew.call
// CHECK:STDOUT:   %.loc47_19.2: %i32 = converted %TooFew.call, %.loc47_19.1
// CHECK:STDOUT:   return %.loc47_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a.param_patt: %i32, %b.param_patt: %i32, %c.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: %i32 = name_ref c, %c
// CHECK:STDOUT:   %TooMany.call: init %i32 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc51_26.1: %i32 = value_of_initializer %TooMany.call
// CHECK:STDOUT:   %.loc51_26.2: %i32 = converted %TooMany.call, %.loc51_26.1
// CHECK:STDOUT:   return %.loc51_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc55_29.1: bool = value_of_initializer %BadReturnType.call
// CHECK:STDOUT:   %.loc55_29.2: bool = converted %BadReturnType.call, %.loc55_29.1
// CHECK:STDOUT:   return %.loc55_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Add.type.1: type = fn_type @Add.1 [template]
// CHECK:STDOUT:   %Add: %Add.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type.1 = fn_decl @Add.1 [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref.loc6: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc6_18: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_18: <bound method> = bound_method %int_2147483647.loc6, %impl.elem0.loc6_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_18: <specific function> = specific_function %Convert.bound.loc6_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init %i32 = call %Convert.specific_fn.loc6_18(%int_2147483647.loc6) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %int.convert_checked.loc6_18 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_2147483647.loc6, %.loc6_18.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc6_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_30: <bound method> = bound_method %int_0, %impl.elem0.loc6_30 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_30: <specific function> = specific_function %Convert.bound.loc6_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_30: init %i32 = call %Convert.specific_fn.loc6_30(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc6_30.1: %i32 = value_of_initializer %int.convert_checked.loc6_30 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc6_30.2: %i32 = converted %int_0, %.loc6_30.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.sadd.loc6: init %i32 = call %Add.ref.loc6(%.loc6_18.2, %.loc6_30.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %int.sadd.loc6 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %int.sadd.loc6, %.loc6_32.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Add.ref.loc10: %Add.type.1 = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_2147483647.loc10: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc10_18: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_18: <bound method> = bound_method %int_2147483647.loc10, %impl.elem0.loc10_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_18: <specific function> = specific_function %Convert.bound.loc10_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %i32 = call %Convert.specific_fn.loc10_18(%int_2147483647.loc10) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc10_18.1: %i32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc10_18.2: %i32 = converted %int_2147483647.loc10, %.loc10_18.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc10_30: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_30: <bound method> = bound_method %int_1, %impl.elem0.loc10_30 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_30: <specific function> = specific_function %Convert.bound.loc10_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc10_30: init %i32 = call %Convert.specific_fn.loc10_30(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %int.convert_checked.loc10_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_1, %.loc10_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.sadd.loc10: init %i32 = call %Add.ref.loc10(%.loc10_18.2, %.loc10_30.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_32.1: %i32 = value_of_initializer %int.sadd.loc10 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_32.2: %i32 = converted %int.sadd.loc10, %.loc10_32.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc10_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
