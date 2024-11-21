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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: error: cannot access member of interface `Ordered` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  return a >= b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Ordered.type: type = facet_type <@Ordered> [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Less.type.1: type = fn_type @Less.1 [template]
// CHECK:STDOUT:   %Less.1: %Less.type.1 = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent.type.1: type = fn_type @LessOrEquivalent.1 [template]
// CHECK:STDOUT:   %LessOrEquivalent.1: %LessOrEquivalent.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Greater.type.1: type = fn_type @Greater.1 [template]
// CHECK:STDOUT:   %Greater.1: %Greater.type.1 = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.1: type = fn_type @GreaterOrEquivalent.1 [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.1: %GreaterOrEquivalent.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Less.type.2: type = fn_type @Less.2 [template]
// CHECK:STDOUT:   %LessOrEquivalent.type.2: type = fn_type @LessOrEquivalent.2 [template]
// CHECK:STDOUT:   %Greater.type.2: type = fn_type @Greater.2 [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.2: type = fn_type @GreaterOrEquivalent.2 [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%Less.1, %LessOrEquivalent.1, %Greater.1, %GreaterOrEquivalent.1) [template]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [template]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [template]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [template]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Ordered = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.11
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+73, loaded [template = constants.%Ordered.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Ordered.ref: type = name_ref Ordered, imports.%import_ref.1 [template = constants.%Ordered.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [template = constants.%TestLess] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc13_16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc13_22: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc13_28.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %bool.make_type, %.loc13_28.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [template = constants.%TestLessEqual] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc17_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc17_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc17_33.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc17_33.2: type = converted %bool.make_type, %.loc17_33.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [template = constants.%TestGreater] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc21_19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc21_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc21_31.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc21_31.2: type = converted %bool.make_type, %.loc21_31.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [template = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc25_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc25_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc25_36.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc25_36.2: type = converted %bool.make_type, %.loc25_36.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %Ordered.ref {
// CHECK:STDOUT:   %Less.decl: %Less.type.1 = fn_decl @Less.1 [template = constants.%Less.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc7_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %bool.make_type, %.loc7_33.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LessOrEquivalent.decl: %LessOrEquivalent.type.1 = fn_decl @LessOrEquivalent.1 [template = constants.%LessOrEquivalent.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc8_39: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_45.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc8_45.2: type = converted %bool.make_type, %.loc8_45.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Greater.decl: %Greater.type.1 = fn_decl @Greater.1 [template = constants.%Greater.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc9_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_36.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc9_36.2: type = converted %bool.make_type, %.loc9_36.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GreaterOrEquivalent.decl: %GreaterOrEquivalent.type.1 = fn_decl @GreaterOrEquivalent.1 [template = constants.%GreaterOrEquivalent.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc10_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc10_42: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc10_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc10_48.2: type = converted %bool.make_type, %.loc10_48.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = interface_witness (%Less.decl, %LessOrEquivalent.decl, %Greater.decl, %GreaterOrEquivalent.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = %Less.decl
// CHECK:STDOUT:   .LessOrEquivalent = %LessOrEquivalent.decl
// CHECK:STDOUT:   .Greater = %Greater.decl
// CHECK:STDOUT:   .GreaterOrEquivalent = %GreaterOrEquivalent.decl
// CHECK:STDOUT:   witness = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc14_12.1: %Less.type.2 = interface_witness_access constants.%.5, element0 [template = constants.%Less.1]
// CHECK:STDOUT:   %.loc14_12.2: <bound method> = bound_method %a.ref, %.loc14_12.1
// CHECK:STDOUT:   %Less.call: init bool = call %.loc14_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_15.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc14_15.2: bool = converted %Less.call, %.loc14_15.1
// CHECK:STDOUT:   return %.loc14_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc18_12.1: %LessOrEquivalent.type.2 = interface_witness_access constants.%.5, element1 [template = constants.%LessOrEquivalent.1]
// CHECK:STDOUT:   %.loc18_12.2: <bound method> = bound_method %a.ref, %.loc18_12.1
// CHECK:STDOUT:   %LessOrEquivalent.call: init bool = call %.loc18_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_16.1: bool = value_of_initializer %LessOrEquivalent.call
// CHECK:STDOUT:   %.loc18_16.2: bool = converted %LessOrEquivalent.call, %.loc18_16.1
// CHECK:STDOUT:   return %.loc18_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc22_12.1: %Greater.type.2 = interface_witness_access constants.%.5, element2 [template = constants.%Greater.1]
// CHECK:STDOUT:   %.loc22_12.2: <bound method> = bound_method %a.ref, %.loc22_12.1
// CHECK:STDOUT:   %Greater.call: init bool = call %.loc22_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_15.1: bool = value_of_initializer %Greater.call
// CHECK:STDOUT:   %.loc22_15.2: bool = converted %Greater.call, %.loc22_15.1
// CHECK:STDOUT:   return %.loc22_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc26_12.1: %GreaterOrEquivalent.type.2 = interface_witness_access constants.%.5, element3 [template = constants.%GreaterOrEquivalent.1]
// CHECK:STDOUT:   %.loc26_12.2: <bound method> = bound_method %a.ref, %.loc26_12.1
// CHECK:STDOUT:   %GreaterOrEquivalent.call: init bool = call %.loc26_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc26_16.1: bool = value_of_initializer %GreaterOrEquivalent.call
// CHECK:STDOUT:   %.loc26_16.2: bool = converted %GreaterOrEquivalent.call, %.loc26_16.1
// CHECK:STDOUT:   return %.loc26_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [template]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [template]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [template]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Ordered = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [template = constants.%TestLess] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc6_16: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc6_22: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc6_28.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc6_28.2: type = converted %bool.make_type, %.loc6_28.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [template = constants.%TestLessEqual] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc14_21: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc14_27: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %bool.make_type, %.loc14_33.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [template = constants.%TestGreater] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc22_19: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc22_25: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc22_31.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc22_31.2: type = converted %bool.make_type, %.loc22_31.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [template = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc30_24: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc30_30: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc30_36.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc30_36.2: type = converted %bool.make_type, %.loc30_36.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
