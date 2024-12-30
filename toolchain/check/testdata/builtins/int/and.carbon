// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/and.carbon

// --- int_and.carbon

library "[[@TEST_NAME]]";

fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(12, 10)];
let arr_p: [i32; 8]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return And(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(And(1, 2)) as Expect(0);
  Test(And(12, 10)) as Expect(8);

  Test(And(1, -1)) as Expect(1);
  Test(And(-2, -3)) as Expect(-4);
  // Ensure the sign bit is treated properly even for 64-bit numbers.
  Test(And(0x7FFF_FFFF_FFFF_FFFF, -3)) as Expect(0x7FFF_FFFF_FFFF_FFFD);
  Test(And(0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";

fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

fn F(a: Core.IntLiteral()) -> Core.IntLiteral() {
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+7]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return AndLit(5, a);
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return AndLit(5, a);
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// Heterogeneous "and" is not supported.
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.and";

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

fn Test(n: Core.IntLiteral()) {
  // OK
  And(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(n, 1);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  And(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(1, n);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  And(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+6]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(n, n);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  And(n, n);
}

// CHECK:STDOUT: --- int_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int_8.2: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_8.2, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
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
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
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
// CHECK:STDOUT: fn @And(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int.and, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %array_type = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0) [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_8) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_8) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_1) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2, %Op.2 [template]
// CHECK:STDOUT:   %int_-2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_3, %Op.2 [template]
// CHECK:STDOUT:   %int_-3: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %int_-4: Core.IntLiteral = int_value -4 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_-4) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_-4) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_4, %Op.2 [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %int_9223372036854775805: Core.IntLiteral = int_value 9223372036854775805 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_9223372036854775805) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_9223372036854775805) [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
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
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
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
// CHECK:STDOUT: fn @And(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.and";
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
// CHECK:STDOUT:   %And.ref.loc10: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.and.loc10: init Core.IntLiteral = call %And.ref.loc10(%int_1.loc10, %int_2.loc10) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.1: Core.IntLiteral = value_of_initializer %int.and.loc10 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc10_17.2: Core.IntLiteral = converted %int.and.loc10, %.loc10_17.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_17.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_17.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_17.4: ref %Expect.2 = temporary %.loc10_17.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %And.ref.loc11: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %int.and.loc11: init Core.IntLiteral = call %And.ref.loc11(%int_12, %int_10) [template = constants.%int_8]
// CHECK:STDOUT:   %.loc11_19.1: Core.IntLiteral = value_of_initializer %int.and.loc11 [template = constants.%int_8]
// CHECK:STDOUT:   %.loc11_19.2: Core.IntLiteral = converted %int.and.loc11, %.loc11_19.1 [template = constants.%int_8]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_8) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_19.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_19.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_8) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_19.4: ref %Expect.3 = temporary %.loc11_19.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %And.ref.loc13: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_1.loc13_12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc13_16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13: <bound method> = bound_method %int_1.loc13_16, %impl.elem0.loc13 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc13: init Core.IntLiteral = call %Op.bound.loc13(%int_1.loc13_16) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_15.2: Core.IntLiteral = converted %int.snegate.loc13, %.loc13_15.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.and.loc13: init Core.IntLiteral = call %And.ref.loc13(%int_1.loc13_12, %.loc13_15.2) [template = constants.%int_1]
// CHECK:STDOUT:   %.loc13_18.1: Core.IntLiteral = value_of_initializer %int.and.loc13 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc13_18.2: Core.IntLiteral = converted %int.and.loc13, %.loc13_18.1 [template = constants.%int_1]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc13_18.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.4 = call %Test.specific_fn.loc13() to %.loc13_18.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc13_30: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc13_18.4: ref %Expect.4 = temporary %.loc13_18.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %And.ref.loc14: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc14_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14_12: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_12 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc14_12: init Core.IntLiteral = call %Op.bound.loc14_12(%int_2.loc14) [template = constants.%int_-2]
// CHECK:STDOUT:   %int_3.loc14: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc14_16: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14_16: <bound method> = bound_method %int_3.loc14, %impl.elem0.loc14_16 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc14_16: init Core.IntLiteral = call %Op.bound.loc14_16(%int_3.loc14) [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc14_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14_12 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc14_12.2: Core.IntLiteral = converted %int.snegate.loc14_12, %.loc14_12.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc14_16.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14_16 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc14_16.2: Core.IntLiteral = converted %int.snegate.loc14_16, %.loc14_16.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %int.and.loc14: init Core.IntLiteral = call %And.ref.loc14(%.loc14_12.2, %.loc14_16.2) [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc14_19.1: Core.IntLiteral = value_of_initializer %int.and.loc14 [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc14_19.2: Core.IntLiteral = converted %int.and.loc14, %.loc14_19.1 [template = constants.%int_-4]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_-4) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc14_19.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.5 = call %Test.specific_fn.loc14() to %.loc14_19.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %impl.elem0.loc14_31: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14_31: <bound method> = bound_method %int_4, %impl.elem0.loc14_31 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc14_31: init Core.IntLiteral = call %Op.bound.loc14_31(%int_4) [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc14_33.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14_31 [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc14_33.2: Core.IntLiteral = converted %int.snegate.loc14_31, %.loc14_33.1 [template = constants.%int_-4]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_-4) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc14_19.4: ref %Expect.5 = temporary %.loc14_19.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %And.ref.loc16: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int_3.loc16: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc16: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16: <bound method> = bound_method %int_3.loc16, %impl.elem0.loc16 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc16: init Core.IntLiteral = call %Op.bound.loc16(%int_3.loc16) [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc16_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc16_35.2: Core.IntLiteral = converted %int.snegate.loc16, %.loc16_35.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %int.and.loc16: init Core.IntLiteral = call %And.ref.loc16(%int_9223372036854775807, %.loc16_35.2) [template = constants.%int_9223372036854775805]
// CHECK:STDOUT:   %.loc16_38.1: Core.IntLiteral = value_of_initializer %int.and.loc16 [template = constants.%int_9223372036854775805]
// CHECK:STDOUT:   %.loc16_38.2: Core.IntLiteral = converted %int.and.loc16, %.loc16_38.1 [template = constants.%int_9223372036854775805]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_9223372036854775805) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc16_38.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.6 = call %Test.specific_fn.loc16() to %.loc16_38.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775805: Core.IntLiteral = int_value 9223372036854775805 [template = constants.%int_9223372036854775805]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_9223372036854775805) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc16_38.4: ref %Expect.6 = temporary %.loc16_38.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %And.ref.loc17: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_9223372036854775808.loc17_12: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc17: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc17: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc17: init Core.IntLiteral = call %Op.bound.loc17(%int_1.loc17) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc17_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc17_35.2: Core.IntLiteral = converted %int.snegate.loc17, %.loc17_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.and.loc17: init Core.IntLiteral = call %And.ref.loc17(%int_9223372036854775808.loc17_12, %.loc17_35.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc17_38.1: Core.IntLiteral = value_of_initializer %int.and.loc17 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc17_38.2: Core.IntLiteral = converted %int.and.loc17, %.loc17_38.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc17_38.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.7 = call %Test.specific_fn.loc17() to %.loc17_38.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc17_50: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc17_38.4: ref %Expect.7 = temporary %.loc17_38.3, %Test.call.loc17
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
// CHECK:STDOUT: specific @Test(constants.%int_8) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_8
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_8
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_8) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_8
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-4) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-4
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-4
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-4) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775805) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775805
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775805
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775805) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775805
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775805
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
// CHECK:STDOUT: --- fail_literal_runtime.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %AndLit.type: type = fn_type @AndLit [template]
// CHECK:STDOUT:   %AndLit: %AndLit.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AndLit = %AndLit.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AndLit.decl: %AndLit.type = fn_decl @AndLit [template = constants.%AndLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_58: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_62: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_74: init type = call %IntLiteral.ref.loc4_62() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.1: type = value_of_initializer %int_literal.make_type.loc4_74 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.2: type = converted %int_literal.make_type.loc4_74, %.loc4_74.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %int_literal.make_type.loc4_30, %.loc4_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_52.1: type = splice_block %.loc4_52.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.2: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.3: type = converted %int_literal.make_type.loc4_52, %.loc4_52.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc6_31: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc6_35: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc6_47: init type = call %IntLiteral.ref.loc6_35() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_47.1: type = value_of_initializer %int_literal.make_type.loc6_47 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_47.2: type = converted %int_literal.make_type.loc6_47, %.loc6_47.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_25.1: type = splice_block %.loc6_25.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc6_9: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc6_13: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc6_25: init type = call %IntLiteral.ref.loc6_13() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_25.2: type = value_of_initializer %int_literal.make_type.loc6_25 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_25.3: type = converted %int_literal.make_type.loc6_25, %.loc6_25.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AndLit(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AndLit.ref: %AndLit.type = name_ref AndLit, file.%AndLit.decl [template = constants.%AndLit]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %a.ref: Core.IntLiteral = name_ref a, %a
// CHECK:STDOUT:   %int.and: init Core.IntLiteral = call %AndLit.ref(%int_5, %a.ref)
// CHECK:STDOUT:   %.loc14_22.1: Core.IntLiteral = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc14_22.2: Core.IntLiteral = converted %int.and, %.loc14_22.1
// CHECK:STDOUT:   return %.loc14_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %MixedAnd1.type: type = fn_type @MixedAnd1 [template]
// CHECK:STDOUT:   %MixedAnd1: %MixedAnd1.type = struct_value () [template]
// CHECK:STDOUT:   %MixedAnd2.type: type = fn_type @MixedAnd2 [template]
// CHECK:STDOUT:   %MixedAnd2: %MixedAnd2.type = struct_value () [template]
// CHECK:STDOUT:   %MixedAnd3.type: type = fn_type @MixedAnd3 [template]
// CHECK:STDOUT:   %MixedAnd3: %MixedAnd3.type = struct_value () [template]
// CHECK:STDOUT:   %MixedAnd4.type: type = fn_type @MixedAnd4 [template]
// CHECK:STDOUT:   %MixedAnd4: %MixedAnd4.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IntLiteral = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MixedAnd1 = %MixedAnd1.decl
// CHECK:STDOUT:     .MixedAnd2 = %MixedAnd2.decl
// CHECK:STDOUT:     .MixedAnd3 = %MixedAnd3.decl
// CHECK:STDOUT:     .MixedAnd4 = %MixedAnd4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MixedAnd1.decl: %MixedAnd1.type = fn_decl @MixedAnd1 [template = constants.%MixedAnd1] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_47: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_47: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc9_17: type = splice_block %i32.loc9_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc9_41.1: type = splice_block %.loc9_41.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_41.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_41.3: type = converted %int_literal.make_type, %.loc9_41.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MixedAnd2.decl: %MixedAnd2.type = fn_decl @MixedAnd2 [template = constants.%MixedAnd2] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_47: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_47: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc14_33.1: type = splice_block %.loc14_33.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc14_33.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc14_33.3: type = converted %int_literal.make_type, %.loc14_33.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc14_39: type = splice_block %i32.loc14_39 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_39: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_39: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MixedAnd3.decl: %MixedAnd3.type = fn_decl @MixedAnd3 [template = constants.%MixedAnd3] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc19_47: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc19_51: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc19_63: init type = call %IntLiteral.ref.loc19_51() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc19_63.1: type = value_of_initializer %int_literal.make_type.loc19_63 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc19_63.2: type = converted %int_literal.make_type.loc19_63, %.loc19_63.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc19_17: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc19_41.1: type = splice_block %.loc19_41.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc19_25: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc19_29: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc19_41: init type = call %IntLiteral.ref.loc19_29() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc19_41.2: type = value_of_initializer %int_literal.make_type.loc19_41 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc19_41.3: type = converted %int_literal.make_type.loc19_41, %.loc19_41.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MixedAnd4.decl: %MixedAnd4.type = fn_decl @MixedAnd4 [template = constants.%MixedAnd4] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc24_47: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc24_51: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc24_63: init type = call %IntLiteral.ref.loc24_51() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc24_63.1: type = value_of_initializer %int_literal.make_type.loc24_63 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc24_63.2: type = converted %int_literal.make_type.loc24_63, %.loc24_63.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc24_33.1: type = splice_block %.loc24_33.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc24_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc24_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc24_33: init type = call %IntLiteral.ref.loc24_21() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc24_33.2: type = value_of_initializer %int_literal.make_type.loc24_33 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc24_33.3: type = converted %int_literal.make_type.loc24_33, %.loc24_33.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc24_39: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MixedAnd1(%a.param_patt: %i32, %b.param_patt: Core.IntLiteral) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MixedAnd2(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MixedAnd3(%a.param_patt: %i32, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MixedAnd4(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> Core.IntLiteral;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_runtime_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_55: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_59: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_71: init type = call %IntLiteral.ref.loc4_59() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.1: type = value_of_initializer %int_literal.make_type.loc4_71 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.2: type = converted %int_literal.make_type.loc4_71, %.loc4_71.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_11: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_15: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_27: init type = call %IntLiteral.ref.loc4_15() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.2: type = value_of_initializer %int_literal.make_type.loc4_27 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.3: type = converted %int_literal.make_type.loc4_27, %.loc4_27.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_49.1: type = splice_block %.loc4_49.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_33: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_37: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_49: init type = call %IntLiteral.ref.loc4_37() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.2: type = value_of_initializer %int_literal.make_type.loc4_49 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.3: type = converted %int_literal.make_type.loc4_49, %.loc4_49.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_28.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_28.3: type = converted %int_literal.make_type, %.loc6_28.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%n.param_patt: Core.IntLiteral) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %And.ref.loc8: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_1.loc8_7: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc8_10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.and.loc8: init Core.IntLiteral = call %And.ref.loc8(%int_1.loc8_7, %int_1.loc8_10) [template = constants.%int_1]
// CHECK:STDOUT:   %And.ref.loc16: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %n.ref.loc16: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.and.loc16: init Core.IntLiteral = call %And.ref.loc16(%n.ref.loc16, %int_1.loc16)
// CHECK:STDOUT:   %And.ref.loc24: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %n.ref.loc24: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.and.loc24: init Core.IntLiteral = call %And.ref.loc24(%int_1.loc24, %n.ref.loc24)
// CHECK:STDOUT:   %And.ref.loc31: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %n.ref.loc31_7: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc31_10: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.and.loc31: init Core.IntLiteral = call %And.ref.loc31(%n.ref.loc31_7, %n.ref.loc31_10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
