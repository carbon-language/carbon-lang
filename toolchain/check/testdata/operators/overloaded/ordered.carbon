// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.OrderedWith(C) {
  fn Less[self: C](other: C) -> bool;
  fn LessOrEquivalent[self: C](other: C) -> bool;
  fn Greater[self: C](other: C) -> bool;
  fn GreaterOrEquivalent[self: C](other: C) -> bool;
}

fn TestLess(a: C, b: C) -> bool {
  return a < b;
}

fn TestLessEqual(a: C, b: C) -> bool {
  return a <= b;
}

fn TestGreater(a: C, b: C) -> bool {
  return a > b;
}

fn TestGreaterEqual(a: C, b: C) -> bool {
  return a >= b;
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestLess(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.OrderedWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a >= b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %OrderedWith.type.107: type = facet_type <@OrderedWith, @OrderedWith(%C)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.ef8: type = fn_type @OrderedWith.Less, @OrderedWith(%C) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.718: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%C) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.394: type = fn_type @OrderedWith.Greater, @OrderedWith(%C) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.f2b: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%C) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness file.%OrderedWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Less.type: type = fn_type @C.as.OrderedWith.impl.Less [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Less: %C.as.OrderedWith.impl.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.LessOrEquivalent.type: type = fn_type @C.as.OrderedWith.impl.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.LessOrEquivalent: %C.as.OrderedWith.impl.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Greater.type: type = fn_type @C.as.OrderedWith.impl.Greater [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Greater: %C.as.OrderedWith.impl.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.GreaterOrEquivalent.type: type = fn_type @C.as.OrderedWith.impl.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %C.as.OrderedWith.impl.GreaterOrEquivalent: %C.as.OrderedWith.impl.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.107 = facet_value %C, (%OrderedWith.impl_witness) [concrete]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [concrete]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [concrete]
// CHECK:STDOUT:   %.440: type = fn_type_with_self_type %OrderedWith.Less.type.ef8, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [concrete]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7af: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.718, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [concrete]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [concrete]
// CHECK:STDOUT:   %.3d4: type = fn_type_with_self_type %OrderedWith.Greater.type.394, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [concrete]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.3b1: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.f2b, %OrderedWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.OrderedWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc6_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %OrderedWith.ref: %OrderedWith.type.270 = name_ref OrderedWith, imports.%Core.OrderedWith [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:     %C.ref.loc6_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %OrderedWith.type: type = facet_type <@OrderedWith, @OrderedWith(constants.%C)> [concrete = constants.%OrderedWith.type.107]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OrderedWith.impl_witness_table = impl_witness_table (@C.as.OrderedWith.impl.%C.as.OrderedWith.impl.Less.decl, @C.as.OrderedWith.impl.%C.as.OrderedWith.impl.LessOrEquivalent.decl, @C.as.OrderedWith.impl.%C.as.OrderedWith.impl.Greater.decl, @C.as.OrderedWith.impl.%C.as.OrderedWith.impl.GreaterOrEquivalent.decl), @C.as.OrderedWith.impl [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness %OrderedWith.impl_witness_table [concrete = constants.%OrderedWith.impl_witness]
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [concrete = constants.%TestLess] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_28.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %Bool.call, %.loc13_28.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc13_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc13_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [concrete = constants.%TestLessEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc17_33.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc17_33.2: type = converted %Bool.call, %.loc17_33.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc17_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc17_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [concrete = constants.%TestGreater] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc21_31.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc21_31.2: type = converted %Bool.call, %.loc21_31.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc21_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc21_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [concrete = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc25_36.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc25_36.2: type = converted %Bool.call, %.loc25_36.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc25_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc25_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.OrderedWith.impl: %C.ref.loc6_6 as %OrderedWith.type {
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Less.decl: %C.as.OrderedWith.impl.Less.type = fn_decl @C.as.OrderedWith.impl.Less [concrete = constants.%C.as.OrderedWith.impl.Less] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %Bool.call, %.loc7_33.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc7_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc7_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.as.OrderedWith.impl.LessOrEquivalent.decl: %C.as.OrderedWith.impl.LessOrEquivalent.type = fn_decl @C.as.OrderedWith.impl.LessOrEquivalent [concrete = constants.%C.as.OrderedWith.impl.LessOrEquivalent] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_45.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc8_45.2: type = converted %Bool.call, %.loc8_45.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc8_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc8_39: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Greater.decl: %C.as.OrderedWith.impl.Greater.type = fn_decl @C.as.OrderedWith.impl.Greater [concrete = constants.%C.as.OrderedWith.impl.Greater] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc9_36.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc9_36.2: type = converted %Bool.call, %.loc9_36.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc9_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc9_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.as.OrderedWith.impl.GreaterOrEquivalent.decl: %C.as.OrderedWith.impl.GreaterOrEquivalent.type = fn_decl @C.as.OrderedWith.impl.GreaterOrEquivalent [concrete = constants.%C.as.OrderedWith.impl.GreaterOrEquivalent] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_48.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc10_48.2: type = converted %Bool.call, %.loc10_48.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc10_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc10_42: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Less = %C.as.OrderedWith.impl.Less.decl
// CHECK:STDOUT:   .LessOrEquivalent = %C.as.OrderedWith.impl.LessOrEquivalent.decl
// CHECK:STDOUT:   .Greater = %C.as.OrderedWith.impl.Greater.decl
// CHECK:STDOUT:   .GreaterOrEquivalent = %C.as.OrderedWith.impl.GreaterOrEquivalent.decl
// CHECK:STDOUT:   witness = file.%OrderedWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.OrderedWith.impl.Less(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.OrderedWith.impl.LessOrEquivalent(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.OrderedWith.impl.Greater(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.OrderedWith.impl.GreaterOrEquivalent(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.440 = impl_witness_access constants.%OrderedWith.impl_witness, element0 [concrete = constants.%C.as.OrderedWith.impl.Less]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Less.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_15.1: bool = value_of_initializer %C.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:   %.loc14_15.2: bool = converted %C.as.OrderedWith.impl.Less.call, %.loc14_15.1
// CHECK:STDOUT:   return %.loc14_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.7af = impl_witness_access constants.%OrderedWith.impl_witness, element1 [concrete = constants.%C.as.OrderedWith.impl.LessOrEquivalent]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %C.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_16.1: bool = value_of_initializer %C.as.OrderedWith.impl.LessOrEquivalent.call
// CHECK:STDOUT:   %.loc18_16.2: bool = converted %C.as.OrderedWith.impl.LessOrEquivalent.call, %.loc18_16.1
// CHECK:STDOUT:   return %.loc18_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.3d4 = impl_witness_access constants.%OrderedWith.impl_witness, element2 [concrete = constants.%C.as.OrderedWith.impl.Greater]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem2
// CHECK:STDOUT:   %C.as.OrderedWith.impl.Greater.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_15.1: bool = value_of_initializer %C.as.OrderedWith.impl.Greater.call
// CHECK:STDOUT:   %.loc22_15.2: bool = converted %C.as.OrderedWith.impl.Greater.call, %.loc22_15.1
// CHECK:STDOUT:   return %.loc22_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.3b1 = impl_witness_access constants.%OrderedWith.impl_witness, element3 [concrete = constants.%C.as.OrderedWith.impl.GreaterOrEquivalent]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem3
// CHECK:STDOUT:   %C.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc26_16.1: bool = value_of_initializer %C.as.OrderedWith.impl.GreaterOrEquivalent.call
// CHECK:STDOUT:   %.loc26_16.2: bool = converted %C.as.OrderedWith.impl.GreaterOrEquivalent.call, %.loc26_16.1
// CHECK:STDOUT:   return %.loc26_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [concrete]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [concrete]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [concrete]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [concrete]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [concrete = constants.%TestLess] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc6_28.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc6_28.2: type = converted %Bool.call, %.loc6_28.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc6_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc6_22: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [concrete = constants.%TestLessEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %Bool.call, %.loc14_33.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc14_21: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc14_27: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [concrete = constants.%TestGreater] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc22_31.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc22_31.2: type = converted %Bool.call, %.loc22_31.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc22_19: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc22_25: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [concrete = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc30_36.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc30_36.2: type = converted %Bool.call, %.loc30_36.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc30_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc30_30: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
