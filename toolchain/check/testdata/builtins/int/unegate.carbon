// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/unegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/unegate.carbon

// --- int_negate.carbon

fn Negate(a: u32) -> u32 = "int.unegate";

var arr: [u32; Negate(Negate(123))];
let arr_p: [u32; 123]* = &arr;

let n: u32 = Negate(1);

fn RuntimeCall(a: u32, b: u32) -> u32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> u32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: u32) -> bool = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: u32) -> bool = "int.unegate";
fn JustRight(a: u32) -> u32 = "int.unegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: [u32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
var too_few: [u32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: [u32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: [u32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: [u32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [u32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: [u32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [u32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: u32) -> u32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> u32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: u32, b: u32, c: u32) -> u32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: u32, b: u32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: u32) -> bool = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return BadReturnType(a, b);
}

// --- overflow.carbon

package Overflow;

fn Negate(a: u32) -> u32 = "int.unegate";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn F() {
  // -(-INT_MAX) is INT_MAX.
  Test(Negate(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  Test(-(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  // -(-(INT_MAX + 1)) is `INT_MAX + 1`.
  Test(Negate(Negate(0x8000_0000))) as Expect(0x8000_0000);
  Test(-(Negate(0x8000_0000))) as Expect(0x8000_0000);
}


// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_123.1: Core.IntLiteral = int_value 123 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%u32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_123.1, %u32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %u32 = int_value 1 [template]
// CHECK:STDOUT:   %int_4294967295: %u32 = int_value 4294967295 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.1
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
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc2_22: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc2: type = splice_block %u32.loc2_14 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc2_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc2_14: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %u32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %u32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc9_35: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc9_19: type = splice_block %u32.loc9_19 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc9_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc9_19: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %u32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc9_27: type = splice_block %u32.loc9_27 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc9_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc9_27: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %u32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %u32) -> %u32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %u32, %b.param_patt: %u32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %int.unegate: init %u32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: %u32 = value_of_initializer %int.unegate
// CHECK:STDOUT:   %.loc10_19.2: %u32 = converted %int.unegate, %.loc10_19.1
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
// CHECK:STDOUT:   %int.convert_checked: init %u32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.1: %u32 = value_of_initializer %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.2: %u32 = converted %int_1, %.loc7_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.unegate: init %u32 = call %Negate.ref(%.loc7_21.2) [template = constants.%int_4294967295]
// CHECK:STDOUT:   %.loc7_23.1: %u32 = value_of_initializer %int.unegate [template = constants.%int_4294967295]
// CHECK:STDOUT:   %.loc7_23.2: %u32 = converted %int.unegate, %.loc7_23.1 [template = constants.%int_4294967295]
// CHECK:STDOUT:   %n: %u32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
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
// CHECK:STDOUT:     .UInt = %import_ref.1
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
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %u32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %u32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc13_31: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_15: type = splice_block %u32.loc13_15 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc13_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_15: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %u32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc13_23: type = splice_block %u32.loc13_23 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_23: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %u32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type, %.loc18_29.1 [template = bool]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_21: type = splice_block %u32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc19_25: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc19: type = splice_block %u32.loc19_17 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc19_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc19_17: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
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
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc46_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc46_33: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc46: type = splice_block %u32.loc46_25 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc46_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc46_25: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %u32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %u32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %u32 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %u32 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc57_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc57_50: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc57_26: type = splice_block %u32.loc57_26 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc57_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc57_26: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %u32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc57_34: type = splice_block %u32.loc57_34 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc57_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc57_34: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %u32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %u32 = value_param runtime_param2
// CHECK:STDOUT:     %.loc57_42: type = splice_block %u32.loc57_42 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc57_42: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc57_42: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %u32 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %u32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %u32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc68_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc68_48.2: type = converted %bool.make_type, %.loc68_48.1 [template = bool]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc68_32: type = splice_block %u32.loc68_32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc68_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc68_32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %u32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc68_40: type = splice_block %u32.loc68_40 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc68_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc68_40: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %u32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> %u32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: %u32, %b.param_patt: %u32) -> %u32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: %u32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: %u32) -> %u32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a.param_patt: %u32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a.param_patt: %u32, %b.param_patt: %u32, %c.param_patt: %u32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %u32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: %u32 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a.param_patt: %u32, %b.param_patt: %u32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %u32 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %N.2: %u32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %u32 = symbolic_binding_pattern N, 0 [symbolic]
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
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%u32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %u32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_2147483649: %u32 = int_value 2147483649 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_2147483647.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_2147483647.2) [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.5, @impl.11(%int_32) [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_2147483649, %Op.14 [template]
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function %Op.bound.1, @Op.5(%int_32) [template]
// CHECK:STDOUT:   %int_2147483648.1: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483648.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483648.2: %u32 = int_value 2147483648 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_2147483648.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_2147483648.2) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2147483648.2, %Op.14 [template]
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function %Op.bound.2, @Op.5(%int_32) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Negate = %import_ref.193
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc4_22: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %u32.loc4_14 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc4_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc4_14: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6: type = splice_block %u32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_14.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %u32 = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc7_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc7_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13: type = splice_block %u32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: %u32) {
// CHECK:STDOUT:   %N.loc6_14.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.2)]
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
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %u32) -> %u32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: %u32) {
// CHECK:STDOUT:   %N.loc7_9.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc7_29.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %u32) -> %return.param_patt: @Test.%Expect.loc7_29.2 (%Expect.1) {
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
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc11_8: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc11_15: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc11_22: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc11_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_22: <bound method> = bound_method %int_2147483647.loc11_22, %impl.elem0.loc11_22 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_22: <specific function> = specific_function %Convert.bound.loc11_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_22: init %u32 = call %Convert.specific_fn.loc11_22(%int_2147483647.loc11_22) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_22.1: %u32 = value_of_initializer %int.convert_checked.loc11_22 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_22.2: %u32 = converted %int_2147483647.loc11_22, %.loc11_22.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.unegate.loc11_33: init %u32 = call %Negate.ref.loc11_15(%.loc11_22.2) [template = constants.%int_2147483649]
// CHECK:STDOUT:   %.loc11_33.1: %u32 = value_of_initializer %int.unegate.loc11_33 [template = constants.%int_2147483649]
// CHECK:STDOUT:   %.loc11_33.2: %u32 = converted %int.unegate.loc11_33, %.loc11_33.1 [template = constants.%int_2147483649]
// CHECK:STDOUT:   %int.unegate.loc11_34: init %u32 = call %Negate.ref.loc11_8(%.loc11_33.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_35.1: %u32 = value_of_initializer %int.unegate.loc11_34 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_35.2: %u32 = converted %int.unegate.loc11_34, %.loc11_35.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_2147483647.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_35.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_35.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc11_47: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc11_58: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_58: <bound method> = bound_method %int_2147483647.loc11_47, %impl.elem0.loc11_58 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_58: <specific function> = specific_function %Convert.bound.loc11_58, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_58: init %u32 = call %Convert.specific_fn.loc11_58(%int_2147483647.loc11_47) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_58.1: %u32 = value_of_initializer %int.convert_checked.loc11_58 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc11_58.2: %u32 = converted %int_2147483647.loc11_47, %.loc11_58.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_2147483647.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_35.4: ref %Expect.2 = temporary %.loc11_35.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc12_17: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc12_17: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_17: <bound method> = bound_method %int_2147483647.loc12_17, %impl.elem0.loc12_17 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_17: <specific function> = specific_function %Convert.bound.loc12_17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_17: init %u32 = call %Convert.specific_fn.loc12_17(%int_2147483647.loc12_17) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_17.1: %u32 = value_of_initializer %int.convert_checked.loc12_17 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_17.2: %u32 = converted %int_2147483647.loc12_17, %.loc12_17.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.unegate.loc12_28: init %u32 = call %Negate.ref.loc12(%.loc12_17.2) [template = constants.%int_2147483649]
// CHECK:STDOUT:   %impl.elem0.loc12_8: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc12: <bound method> = bound_method %int.unegate.loc12_28, %impl.elem0.loc12_8 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %Op.specific_fn.loc12: <specific function> = specific_function %Op.bound.loc12, @Op.5(constants.%int_32) [template = constants.%Op.specific_fn.1]
// CHECK:STDOUT:   %.loc12_28.1: %u32 = value_of_initializer %int.unegate.loc12_28 [template = constants.%int_2147483649]
// CHECK:STDOUT:   %.loc12_28.2: %u32 = converted %int.unegate.loc12_28, %.loc12_28.1 [template = constants.%int_2147483649]
// CHECK:STDOUT:   %int.unegate.loc12_8: init %u32 = call %Op.specific_fn.loc12(%.loc12_28.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_30.1: %u32 = value_of_initializer %int.unegate.loc12_8 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_30.2: %u32 = converted %int.unegate.loc12_8, %.loc12_30.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_2147483647.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_30.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_30.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483647.loc12_42: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc12_53: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_53: <bound method> = bound_method %int_2147483647.loc12_42, %impl.elem0.loc12_53 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_53: <specific function> = specific_function %Convert.bound.loc12_53, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_53: init %u32 = call %Convert.specific_fn.loc12_53(%int_2147483647.loc12_42) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_53.1: %u32 = value_of_initializer %int.convert_checked.loc12_53 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_53.2: %u32 = converted %int_2147483647.loc12_42, %.loc12_53.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_2147483647.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_30.4: ref %Expect.2 = temporary %.loc12_30.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc14_8: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc14_15: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483648.loc14_22: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc14_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_22: <bound method> = bound_method %int_2147483648.loc14_22, %impl.elem0.loc14_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_22: <specific function> = specific_function %Convert.bound.loc14_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_22: init %u32 = call %Convert.specific_fn.loc14_22(%int_2147483648.loc14_22) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_22.1: %u32 = value_of_initializer %int.convert_checked.loc14_22 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_22.2: %u32 = converted %int_2147483648.loc14_22, %.loc14_22.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %int.unegate.loc14_33: init %u32 = call %Negate.ref.loc14_15(%.loc14_22.2) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_33.1: %u32 = value_of_initializer %int.unegate.loc14_33 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_33.2: %u32 = converted %int.unegate.loc14_33, %.loc14_33.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %int.unegate.loc14_34: init %u32 = call %Negate.ref.loc14_8(%.loc14_33.2) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_35.1: %u32 = value_of_initializer %int.unegate.loc14_34 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_35.2: %u32 = converted %int.unegate.loc14_34, %.loc14_35.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_2147483648.2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc14_35.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.3 = call %Test.specific_fn.loc14() to %.loc14_35.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483648.loc14_47: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc14_58: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_58: <bound method> = bound_method %int_2147483648.loc14_47, %impl.elem0.loc14_58 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_58: <specific function> = specific_function %Convert.bound.loc14_58, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_58: init %u32 = call %Convert.specific_fn.loc14_58(%int_2147483648.loc14_47) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_58.1: %u32 = value_of_initializer %int.convert_checked.loc14_58 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc14_58.2: %u32 = converted %int_2147483648.loc14_47, %.loc14_58.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_2147483648.2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc14_35.4: ref %Expect.3 = temporary %.loc14_35.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Negate.ref.loc15: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483648.loc15_17: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc15_17: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_17: <bound method> = bound_method %int_2147483648.loc15_17, %impl.elem0.loc15_17 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_17: <specific function> = specific_function %Convert.bound.loc15_17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_17: init %u32 = call %Convert.specific_fn.loc15_17(%int_2147483648.loc15_17) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_17.1: %u32 = value_of_initializer %int.convert_checked.loc15_17 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_17.2: %u32 = converted %int_2147483648.loc15_17, %.loc15_17.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %int.unegate.loc15_28: init %u32 = call %Negate.ref.loc15(%.loc15_17.2) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %impl.elem0.loc15_8: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int.unegate.loc15_28, %impl.elem0.loc15_8 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %Op.specific_fn.loc15: <specific function> = specific_function %Op.bound.loc15, @Op.5(constants.%int_32) [template = constants.%Op.specific_fn.2]
// CHECK:STDOUT:   %.loc15_28.1: %u32 = value_of_initializer %int.unegate.loc15_28 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_28.2: %u32 = converted %int.unegate.loc15_28, %.loc15_28.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %int.unegate.loc15_8: init %u32 = call %Op.specific_fn.loc15(%.loc15_28.2) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_30.1: %u32 = value_of_initializer %int.unegate.loc15_8 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_30.2: %u32 = converted %int.unegate.loc15_8, %.loc15_30.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_2147483648.2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc15_30.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.3 = call %Test.specific_fn.loc15() to %.loc15_30.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483648.loc15_42: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc15_53: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_53: <bound method> = bound_method %int_2147483648.loc15_42, %impl.elem0.loc15_53 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_53: <specific function> = specific_function %Convert.bound.loc15_53, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_53: init %u32 = call %Convert.specific_fn.loc15_53(%int_2147483648.loc15_42) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_53.1: %u32 = value_of_initializer %int.convert_checked.loc15_53 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc15_53.2: %u32 = converted %int_2147483648.loc15_42, %.loc15_53.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_2147483648.2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc15_30.4: ref %Expect.3 = temporary %.loc15_30.3, %Test.call.loc15
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @Test(constants.%int_2147483647.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483647.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483647.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483648.2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483648.2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483648.2
// CHECK:STDOUT:   %Expect.loc7_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483648.2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483648.2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483648.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
