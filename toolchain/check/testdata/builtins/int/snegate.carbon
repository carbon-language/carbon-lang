// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/snegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/snegate.carbon

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.snegate";

var arr: [i32; Negate(Negate(123))];
let arr_p: [i32; 123]* = &arr;

let n: i32 = Negate(1);

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.snegate";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Negate(0)) as Expect(0);
  Test(Negate(1)) as Expect(0 - 1);
  Test(Negate(0 - 0x8000_0000_0000_0000)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.snegate";
fn JustRight(a: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: [i32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Negate(a: i32) -> i32 = "int.snegate";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFF_FFFF));

// -INT_MIN is too large for i32.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in negation of -2147483648 [CompileTimeIntegerNegateOverflow]
// CHECK:STDERR: let b: i32 = Negate(-0x8000_0000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
let b: i32 = Negate(-0x8000_0000);

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_123.1: Core.IntLiteral = int_value 123 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_123.1, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc2: type = splice_block %i32.loc2_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc2_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc2_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc9_19: type = splice_block %i32.loc9_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc9_27: type = splice_block %i32.loc9_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int.snegate, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %array_type = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %addr
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.2: %i32 = converted %int_1, %.loc7_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc7_21.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc7_23.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc7_23.2: %i32 = converted %int.snegate, %.loc7_23.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
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
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_-1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_-1) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_0, %Op.2 [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Sub = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.1: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.2: type = converted %int_literal.make_type.loc4_52, %.loc4_52.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %int_literal.make_type.loc4_30, %.loc4_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
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
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.snegate";
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
// CHECK:STDOUT:   %Negate.ref.loc10: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_0.loc10_15: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.snegate.loc10: init Core.IntLiteral = call %Negate.ref.loc10(%int_0.loc10_15) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.1: Core.IntLiteral = value_of_initializer %int.snegate.loc10 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.2: Core.IntLiteral = converted %int.snegate.loc10, %.loc10_17.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc11: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc11_15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.snegate.loc11: init Core.IntLiteral = call %Negate.ref.loc11(%int_1.loc11_15) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_17.1: Core.IntLiteral = value_of_initializer %int.snegate.loc11 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_17.2: Core.IntLiteral = converted %int.snegate.loc11, %.loc11_17.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_17.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc11_33: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc11: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc11: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.ssub.loc11: init Core.IntLiteral = call %Op.bound.loc11(%int_0.loc11, %int_1.loc11_33) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_34.1: Core.IntLiteral = value_of_initializer %int.ssub.loc11 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_34.2: Core.IntLiteral = converted %int.ssub.loc11, %.loc11_34.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_17.4: ref %Expect.3 = temporary %.loc11_17.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_9223372036854775808.loc12_19: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc12: <bound method> = bound_method %int_0.loc12, %impl.elem0.loc12 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.ssub.loc12: init Core.IntLiteral = call %Op.bound.loc12(%int_0.loc12, %int_9223372036854775808.loc12_19) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc12_17.1: Core.IntLiteral = value_of_initializer %int.ssub.loc12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc12_17.2: Core.IntLiteral = converted %int.ssub.loc12, %.loc12_17.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.snegate.loc12: init Core.IntLiteral = call %Negate.ref.loc12(%.loc12_17.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc12_41.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc12_41.2: Core.IntLiteral = converted %int.snegate.loc12, %.loc12_41.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc12_41.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.4 = call %Test.specific_fn.loc12() to %.loc12_41.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc12_53: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc12_41.4: ref %Expect.4 = temporary %.loc12_41.3, %Test.call.loc12
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
// CHECK:STDOUT: specific @Test(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775808
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
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_31: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_15: type = splice_block %i32.loc13_15 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc13_23: type = splice_block %i32.loc13_23 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_23: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type, %.loc18_29.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_21: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc19: type = splice_block %i32.loc19_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc19_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc46_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc46_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc46: type = splice_block %i32.loc46_25 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc46_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc46_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %int_32.loc57_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc57_50: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc57_26: type = splice_block %i32.loc57_26 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc57_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc57_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc57_34: type = splice_block %i32.loc57_34 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc57_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc57_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param runtime_param2
// CHECK:STDOUT:     %.loc57_42: type = splice_block %i32.loc57_42 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc57_42: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc57_42: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc68_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc68_48.2: type = converted %bool.make_type, %.loc68_48.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc68_32: type = splice_block %i32.loc68_32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc68_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc68_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc68_40: type = splice_block %i32.loc68_40 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc68_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc68_40: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: %i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a.param_patt: %i32, %b.param_patt: %i32, %c.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: %i32 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Sub.type.1: type = fn_type @Sub.1 [template]
// CHECK:STDOUT:   %Sub: %Sub.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_-2147483647: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_2147483648, %Op.14 [template]
// CHECK:STDOUT:   %int_-2147483648.1: Core.IntLiteral = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_-2147483648.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-2147483648.2: %i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %i32.loc4_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: %Sub.type.1 = fn_decl @Sub.1 [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %i32.loc5_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc5_19: type = splice_block %i32.loc5_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc8_14: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc8_21: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_2147483647, %impl.elem0.loc8 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_2147483647) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_28.1: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_28.2: %i32 = converted %int_2147483647, %.loc8_28.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.snegate.loc8_39: init %i32 = call %Negate.ref.loc8_21(%.loc8_28.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc8_39.1: %i32 = value_of_initializer %int.snegate.loc8_39 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc8_39.2: %i32 = converted %int.snegate.loc8_39, %.loc8_39.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %int.snegate.loc8_40: init %i32 = call %Negate.ref.loc8_14(%.loc8_39.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_41.1: %i32 = value_of_initializer %int.snegate.loc8_40 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_41.2: %i32 = converted %int.snegate.loc8_40, %.loc8_41.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc8_41.2
// CHECK:STDOUT:   %Negate.ref.loc14: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %impl.elem0.loc14_21.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_2147483648, %impl.elem0.loc14_21.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc14_21: init Core.IntLiteral = call %Op.bound(%int_2147483648) [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc14_21.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int.snegate.loc14_21, %impl.elem0.loc14_21.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc14_21.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14_21 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %.loc14_21.2: Core.IntLiteral = converted %int.snegate.loc14_21, %.loc14_21.1 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.specific_fn.loc14(%.loc14_21.2) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc14_21.3: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc14_21.4: %i32 = converted %int.snegate.loc14_21, %.loc14_21.3 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %int.snegate.loc14_33: init %i32 = call %Negate.ref.loc14(%.loc14_21.4) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.snegate.loc14_33 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int.snegate.loc14_33, %.loc14_34.1 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc14_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
