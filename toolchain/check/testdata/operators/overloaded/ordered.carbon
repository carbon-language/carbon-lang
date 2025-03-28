// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/ordered.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.Ordered {
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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a >= b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Ordered.type: type = facet_type <@Ordered> [concrete]
// CHECK:STDOUT:   %Less.type.341: type = fn_type @Less.1 [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type.859: type = fn_type @LessOrEquivalent.1 [concrete]
// CHECK:STDOUT:   %Greater.type.270: type = fn_type @Greater.1 [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.8af: type = fn_type @GreaterOrEquivalent.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.d54.%Less.decl, @impl.d54.%LessOrEquivalent.decl, @impl.d54.%Greater.decl, @impl.d54.%GreaterOrEquivalent.decl) [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Less.type.25a: type = fn_type @Less.2 [concrete]
// CHECK:STDOUT:   %Less.738: %Less.type.25a = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type.478: type = fn_type @LessOrEquivalent.2 [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.010: %LessOrEquivalent.type.478 = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type.0a1: type = fn_type @Greater.2 [concrete]
// CHECK:STDOUT:   %Greater.afe: %Greater.type.0a1 = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.caf: type = fn_type @GreaterOrEquivalent.2 [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.03b: %GreaterOrEquivalent.type.caf = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.facet: %Ordered.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [concrete]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a6a: type = fn_type_with_self_type %Less.type.341, %Ordered.facet [concrete]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [concrete]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %LessOrEquivalent.type.859, %Ordered.facet [concrete]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [concrete]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [concrete]
// CHECK:STDOUT:   %.0c7: type = fn_type_with_self_type %Greater.type.270, %Ordered.facet [concrete]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [concrete]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a45: type = fn_type_with_self_type %GreaterOrEquivalent.type.8af, %Ordered.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Ordered = %Core.Ordered
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Ordered: type = import_ref Core//prelude/operators/comparison, Ordered, loaded [concrete = constants.%Ordered.type]
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
// CHECK:STDOUT:   impl_decl @impl.d54 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Ordered.ref: type = name_ref Ordered, imports.%Core.Ordered [concrete = constants.%Ordered.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.d54.%Less.decl, @impl.d54.%LessOrEquivalent.decl, @impl.d54.%Greater.decl, @impl.d54.%GreaterOrEquivalent.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [concrete = constants.%TestLess] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_28.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %bool.make_type, %.loc13_28.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc17_33.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc17_33.2: type = converted %bool.make_type, %.loc17_33.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc21_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc21_31.2: type = converted %bool.make_type, %.loc21_31.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc25_36.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc25_36.2: type = converted %bool.make_type, %.loc25_36.1 [concrete = bool]
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
// CHECK:STDOUT: impl @impl.d54: %C.ref as %Ordered.ref {
// CHECK:STDOUT:   %Less.decl: %Less.type.25a = fn_decl @Less.2 [concrete = constants.%Less.738] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %bool.make_type, %.loc7_33.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc7_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc7_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LessOrEquivalent.decl: %LessOrEquivalent.type.478 = fn_decl @LessOrEquivalent.2 [concrete = constants.%LessOrEquivalent.010] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_45.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc8_45.2: type = converted %bool.make_type, %.loc8_45.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc8_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc8_39: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Greater.decl: %Greater.type.0a1 = fn_decl @Greater.2 [concrete = constants.%Greater.afe] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc9_36.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc9_36.2: type = converted %bool.make_type, %.loc9_36.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc9_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc9_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GreaterOrEquivalent.decl: %GreaterOrEquivalent.type.caf = fn_decl @GreaterOrEquivalent.2 [concrete = constants.%GreaterOrEquivalent.03b] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_48.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc10_48.2: type = converted %bool.make_type, %.loc10_48.1 [concrete = bool]
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
// CHECK:STDOUT:   .Less = %Less.decl
// CHECK:STDOUT:   .LessOrEquivalent = %LessOrEquivalent.decl
// CHECK:STDOUT:   .Greater = %Greater.decl
// CHECK:STDOUT:   .GreaterOrEquivalent = %GreaterOrEquivalent.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less.2[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent.2[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater.2[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent.2[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.a6a = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Less.738]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Less.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_15.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc14_15.2: bool = converted %Less.call, %.loc14_15.1
// CHECK:STDOUT:   return %.loc14_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.e16 = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%LessOrEquivalent.010]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %LessOrEquivalent.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_16.1: bool = value_of_initializer %LessOrEquivalent.call
// CHECK:STDOUT:   %.loc18_16.2: bool = converted %LessOrEquivalent.call, %.loc18_16.1
// CHECK:STDOUT:   return %.loc18_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.0c7 = impl_witness_access constants.%impl_witness, element2 [concrete = constants.%Greater.afe]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem2
// CHECK:STDOUT:   %Greater.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_15.1: bool = value_of_initializer %Greater.call
// CHECK:STDOUT:   %.loc22_15.2: bool = converted %Greater.call, %.loc22_15.1
// CHECK:STDOUT:   return %.loc22_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.a45 = impl_witness_access constants.%impl_witness, element3 [concrete = constants.%GreaterOrEquivalent.03b]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem3
// CHECK:STDOUT:   %GreaterOrEquivalent.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc26_16.1: bool = value_of_initializer %GreaterOrEquivalent.call
// CHECK:STDOUT:   %.loc26_16.2: bool = converted %GreaterOrEquivalent.call, %.loc26_16.1
// CHECK:STDOUT:   return %.loc26_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [concrete]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Ordered = %Core.Ordered
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc6_28.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc6_28.2: type = converted %bool.make_type, %.loc6_28.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %bool.make_type, %.loc14_33.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc22_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc22_31.2: type = converted %bool.make_type, %.loc22_31.1 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc30_36.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc30_36.2: type = converted %bool.make_type, %.loc30_36.1 [concrete = bool]
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
