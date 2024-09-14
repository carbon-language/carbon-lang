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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface `Ordered` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface `Ordered` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface `Ordered` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface `Ordered` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  return a >= b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
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
// CHECK:STDOUT:   %Less.2: %Less.type.2 = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent.type.2: type = fn_type @LessOrEquivalent.2 [template]
// CHECK:STDOUT:   %LessOrEquivalent.2: %LessOrEquivalent.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Greater.type.2: type = fn_type @Greater.2 [template]
// CHECK:STDOUT:   %Greater.2: %Greater.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.2: type = fn_type @GreaterOrEquivalent.2 [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.2: %GreaterOrEquivalent.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Less.1, %LessOrEquivalent.1, %Greater.1, %GreaterOrEquivalent.1) [template]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %Less.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.2, %LessOrEquivalent.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element1, imports.%import_ref.13 [template]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.2, %Greater.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element2, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %.2, %GreaterOrEquivalent.type.2 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element3, imports.%import_ref.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Ordered = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.11
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+54, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/comparison, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.3: %.6 = import_ref Core//prelude/operators/comparison, inst+77, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: %.8 = import_ref Core//prelude/operators/comparison, inst+98, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.5: %.10 = import_ref Core//prelude/operators/comparison, inst+119, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.6: %.12 = import_ref Core//prelude/operators/comparison, inst+140, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.7: %Less.type.2 = import_ref Core//prelude/operators/comparison, inst+72, loaded [template = constants.%Less.2]
// CHECK:STDOUT:   %import_ref.8: %LessOrEquivalent.type.2 = import_ref Core//prelude/operators/comparison, inst+93, loaded [template = constants.%LessOrEquivalent.2]
// CHECK:STDOUT:   %import_ref.9: %Greater.type.2 = import_ref Core//prelude/operators/comparison, inst+114, loaded [template = constants.%Greater.2]
// CHECK:STDOUT:   %import_ref.10: %GreaterOrEquivalent.type.2 = import_ref Core//prelude/operators/comparison, inst+135, loaded [template = constants.%GreaterOrEquivalent.2]
// CHECK:STDOUT:   %import_ref.11: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/comparison, inst+72, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/comparison, inst+93, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/comparison, inst+114, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/comparison, inst+135, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Ordered.ref: type = name_ref Ordered, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [template = constants.%TestLess] {
// CHECK:STDOUT:     %C.ref.loc13_16: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc13_13.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestLess.%a: %C = bind_name a, %a.loc13_13.1
// CHECK:STDOUT:     %C.ref.loc13_22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc13_19.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestLess.%b: %C = bind_name b, %b.loc13_19.1
// CHECK:STDOUT:     %bool.make_type.loc13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc13_28.1: type = value_of_initializer %bool.make_type.loc13 [template = bool]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %bool.make_type.loc13, %.loc13_28.1 [template = bool]
// CHECK:STDOUT:     @TestLess.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [template = constants.%TestLessEqual] {
// CHECK:STDOUT:     %C.ref.loc17_21: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc17_18.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestLessEqual.%a: %C = bind_name a, %a.loc17_18.1
// CHECK:STDOUT:     %C.ref.loc17_27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc17_24.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestLessEqual.%b: %C = bind_name b, %b.loc17_24.1
// CHECK:STDOUT:     %bool.make_type.loc17: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc17_33.1: type = value_of_initializer %bool.make_type.loc17 [template = bool]
// CHECK:STDOUT:     %.loc17_33.2: type = converted %bool.make_type.loc17, %.loc17_33.1 [template = bool]
// CHECK:STDOUT:     @TestLessEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [template = constants.%TestGreater] {
// CHECK:STDOUT:     %C.ref.loc21_19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc21_16.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestGreater.%a: %C = bind_name a, %a.loc21_16.1
// CHECK:STDOUT:     %C.ref.loc21_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc21_22.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestGreater.%b: %C = bind_name b, %b.loc21_22.1
// CHECK:STDOUT:     %bool.make_type.loc21: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc21_31.1: type = value_of_initializer %bool.make_type.loc21 [template = bool]
// CHECK:STDOUT:     %.loc21_31.2: type = converted %bool.make_type.loc21, %.loc21_31.1 [template = bool]
// CHECK:STDOUT:     @TestGreater.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [template = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %C.ref.loc25_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc25_21.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestGreaterEqual.%a: %C = bind_name a, %a.loc25_21.1
// CHECK:STDOUT:     %C.ref.loc25_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc25_27.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestGreaterEqual.%b: %C = bind_name b, %b.loc25_27.1
// CHECK:STDOUT:     %bool.make_type.loc25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc25_36.1: type = value_of_initializer %bool.make_type.loc25 [template = bool]
// CHECK:STDOUT:     %.loc25_36.2: type = converted %bool.make_type.loc25, %.loc25_36.1 [template = bool]
// CHECK:STDOUT:     @TestGreaterEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Less = imports.%import_ref.3
// CHECK:STDOUT:   .LessOrEquivalent = imports.%import_ref.4
// CHECK:STDOUT:   .Greater = imports.%import_ref.5
// CHECK:STDOUT:   .GreaterOrEquivalent = imports.%import_ref.6
// CHECK:STDOUT:   witness = (imports.%import_ref.7, imports.%import_ref.8, imports.%import_ref.9, imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.2 {
// CHECK:STDOUT:   %Less.decl: %Less.type.1 = fn_decl @Less.1 [template = constants.%Less.1] {
// CHECK:STDOUT:     %C.ref.loc7_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc7_11.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc7_11.2: %C = bind_name self, %self.loc7_11.1
// CHECK:STDOUT:     %C.ref.loc7_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc7_20.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc7_20.2: %C = bind_name other, %other.loc7_20.1
// CHECK:STDOUT:     %bool.make_type.loc7: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %bool.make_type.loc7 [template = bool]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %bool.make_type.loc7, %.loc7_33.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc7: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LessOrEquivalent.decl: %LessOrEquivalent.type.1 = fn_decl @LessOrEquivalent.1 [template = constants.%LessOrEquivalent.1] {
// CHECK:STDOUT:     %C.ref.loc8_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_23.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc8_23.2: %C = bind_name self, %self.loc8_23.1
// CHECK:STDOUT:     %C.ref.loc8_39: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc8_32.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc8_32.2: %C = bind_name other, %other.loc8_32.1
// CHECK:STDOUT:     %bool.make_type.loc8: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_45.1: type = value_of_initializer %bool.make_type.loc8 [template = bool]
// CHECK:STDOUT:     %.loc8_45.2: type = converted %bool.make_type.loc8, %.loc8_45.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Greater.decl: %Greater.type.1 = fn_decl @Greater.1 [template = constants.%Greater.1] {
// CHECK:STDOUT:     %C.ref.loc9_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_14.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc9_14.2: %C = bind_name self, %self.loc9_14.1
// CHECK:STDOUT:     %C.ref.loc9_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_23.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc9_23.2: %C = bind_name other, %other.loc9_23.1
// CHECK:STDOUT:     %bool.make_type.loc9: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_36.1: type = value_of_initializer %bool.make_type.loc9 [template = bool]
// CHECK:STDOUT:     %.loc9_36.2: type = converted %bool.make_type.loc9, %.loc9_36.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GreaterOrEquivalent.decl: %GreaterOrEquivalent.type.1 = fn_decl @GreaterOrEquivalent.1 [template = constants.%GreaterOrEquivalent.1] {
// CHECK:STDOUT:     %C.ref.loc10_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc10_26.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc10_26.2: %C = bind_name self, %self.loc10_26.1
// CHECK:STDOUT:     %C.ref.loc10_42: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc10_35.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc10_35.2: %C = bind_name other, %other.loc10_35.1
// CHECK:STDOUT:     %bool.make_type.loc10: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc10_48.1: type = value_of_initializer %bool.make_type.loc10 [template = bool]
// CHECK:STDOUT:     %.loc10_48.2: type = converted %bool.make_type.loc10, %.loc10_48.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = interface_witness (%Less.decl, %LessOrEquivalent.decl, %Greater.decl, %GreaterOrEquivalent.decl) [template = constants.%.4]
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
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less.1[@impl.%self.loc7_11.2: %C](@impl.%other.loc7_20.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent.1[@impl.%self.loc8_23.2: %C](@impl.%other.loc8_32.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater.1[@impl.%self.loc9_14.2: %C](@impl.%other.loc9_23.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent.1[@impl.%self.loc10_26.2: %C](@impl.%other.loc10_35.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Less.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Less.2.%Self (%Self)](%other: @Less.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @LessOrEquivalent.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @LessOrEquivalent.2.%Self (%Self)](%other: @LessOrEquivalent.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Greater.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Greater.2.%Self (%Self)](%other: @Greater.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GreaterOrEquivalent.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @GreaterOrEquivalent.2.%Self (%Self)](%other: @GreaterOrEquivalent.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Less.ref: %.6 = name_ref Less, imports.%import_ref.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_12.1: %Less.type.2 = interface_witness_access @impl.%.loc6, element0 [template = constants.%Less.1]
// CHECK:STDOUT:   %.loc14_12.2: <bound method> = bound_method %a.ref, %.loc14_12.1
// CHECK:STDOUT:   %Less.call: init bool = call %.loc14_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_15.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc14_15.2: bool = converted %Less.call, %.loc14_15.1
// CHECK:STDOUT:   return %.loc14_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %LessOrEquivalent.ref: %.8 = name_ref LessOrEquivalent, imports.%import_ref.4 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_12.1: %LessOrEquivalent.type.2 = interface_witness_access @impl.%.loc6, element1 [template = constants.%LessOrEquivalent.1]
// CHECK:STDOUT:   %.loc18_12.2: <bound method> = bound_method %a.ref, %.loc18_12.1
// CHECK:STDOUT:   %LessOrEquivalent.call: init bool = call %.loc18_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_16.1: bool = value_of_initializer %LessOrEquivalent.call
// CHECK:STDOUT:   %.loc18_16.2: bool = converted %LessOrEquivalent.call, %.loc18_16.1
// CHECK:STDOUT:   return %.loc18_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Greater.ref: %.10 = name_ref Greater, imports.%import_ref.5 [template = constants.%.11]
// CHECK:STDOUT:   %.loc22_12.1: %Greater.type.2 = interface_witness_access @impl.%.loc6, element2 [template = constants.%Greater.1]
// CHECK:STDOUT:   %.loc22_12.2: <bound method> = bound_method %a.ref, %.loc22_12.1
// CHECK:STDOUT:   %Greater.call: init bool = call %.loc22_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_15.1: bool = value_of_initializer %Greater.call
// CHECK:STDOUT:   %.loc22_15.2: bool = converted %Greater.call, %.loc22_15.1
// CHECK:STDOUT:   return %.loc22_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %GreaterOrEquivalent.ref: %.12 = name_ref GreaterOrEquivalent, imports.%import_ref.6 [template = constants.%.13]
// CHECK:STDOUT:   %.loc26_12.1: %GreaterOrEquivalent.type.2 = interface_witness_access @impl.%.loc6, element3 [template = constants.%GreaterOrEquivalent.1]
// CHECK:STDOUT:   %.loc26_12.2: <bound method> = bound_method %a.ref, %.loc26_12.1
// CHECK:STDOUT:   %GreaterOrEquivalent.call: init bool = call %.loc26_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc26_16.1: bool = value_of_initializer %GreaterOrEquivalent.call
// CHECK:STDOUT:   %.loc26_16.2: bool = converted %GreaterOrEquivalent.call, %.loc26_16.1
// CHECK:STDOUT:   return %.loc26_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Less.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Less.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LessOrEquivalent.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LessOrEquivalent.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Greater.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Greater.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GreaterOrEquivalent.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GreaterOrEquivalent.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %TestLess.type: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %TestLess: %TestLess.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [template]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Less.type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %TestLessEqual.type: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %TestLessEqual: %TestLessEqual.type = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [template]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %LessOrEquivalent.type [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element1, imports.%import_ref.13 [template]
// CHECK:STDOUT:   %TestGreater.type: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %TestGreater: %TestGreater.type = struct_value () [template]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [template]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.4, %Greater.type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element2, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %TestGreaterEqual.type: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %TestGreaterEqual: %TestGreaterEqual.type = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [template]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.4, %GreaterOrEquivalent.type [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element3, imports.%import_ref.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Ordered = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/comparison, inst+54, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/comparison, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.4: %.5 = import_ref Core//prelude/operators/comparison, inst+77, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.5: %.7 = import_ref Core//prelude/operators/comparison, inst+98, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.6: %.9 = import_ref Core//prelude/operators/comparison, inst+119, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.7: %.11 = import_ref Core//prelude/operators/comparison, inst+140, loaded [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+72, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/comparison, inst+93, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/comparison, inst+114, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/comparison, inst+135, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/comparison, inst+72, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/comparison, inst+93, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/comparison, inst+114, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/comparison, inst+135, unloaded
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
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestLess.decl: %TestLess.type = fn_decl @TestLess [template = constants.%TestLess] {
// CHECK:STDOUT:     %D.ref.loc6_16: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc6_13.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestLess.%a: %D = bind_name a, %a.loc6_13.1
// CHECK:STDOUT:     %D.ref.loc6_22: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc6_19.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestLess.%b: %D = bind_name b, %b.loc6_19.1
// CHECK:STDOUT:     %bool.make_type.loc6: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc6_28.1: type = value_of_initializer %bool.make_type.loc6 [template = bool]
// CHECK:STDOUT:     %.loc6_28.2: type = converted %bool.make_type.loc6, %.loc6_28.1 [template = bool]
// CHECK:STDOUT:     @TestLess.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLessEqual.decl: %TestLessEqual.type = fn_decl @TestLessEqual [template = constants.%TestLessEqual] {
// CHECK:STDOUT:     %D.ref.loc14_21: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc14_18.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestLessEqual.%a: %D = bind_name a, %a.loc14_18.1
// CHECK:STDOUT:     %D.ref.loc14_27: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc14_24.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestLessEqual.%b: %D = bind_name b, %b.loc14_24.1
// CHECK:STDOUT:     %bool.make_type.loc14: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %bool.make_type.loc14 [template = bool]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %bool.make_type.loc14, %.loc14_33.1 [template = bool]
// CHECK:STDOUT:     @TestLessEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreater.decl: %TestGreater.type = fn_decl @TestGreater [template = constants.%TestGreater] {
// CHECK:STDOUT:     %D.ref.loc22_19: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc22_16.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestGreater.%a: %D = bind_name a, %a.loc22_16.1
// CHECK:STDOUT:     %D.ref.loc22_25: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc22_22.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestGreater.%b: %D = bind_name b, %b.loc22_22.1
// CHECK:STDOUT:     %bool.make_type.loc22: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc22_31.1: type = value_of_initializer %bool.make_type.loc22 [template = bool]
// CHECK:STDOUT:     %.loc22_31.2: type = converted %bool.make_type.loc22, %.loc22_31.1 [template = bool]
// CHECK:STDOUT:     @TestGreater.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGreaterEqual.decl: %TestGreaterEqual.type = fn_decl @TestGreaterEqual [template = constants.%TestGreaterEqual] {
// CHECK:STDOUT:     %D.ref.loc30_24: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc30_21.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestGreaterEqual.%a: %D = bind_name a, %a.loc30_21.1
// CHECK:STDOUT:     %D.ref.loc30_30: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc30_27.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestGreaterEqual.%b: %D = bind_name b, %b.loc30_27.1
// CHECK:STDOUT:     %bool.make_type.loc30: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc30_36.1: type = value_of_initializer %bool.make_type.loc30 [template = bool]
// CHECK:STDOUT:     %.loc30_36.2: type = converted %bool.make_type.loc30, %.loc30_36.1 [template = bool]
// CHECK:STDOUT:     @TestGreaterEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Less = imports.%import_ref.4
// CHECK:STDOUT:   .LessOrEquivalent = imports.%import_ref.5
// CHECK:STDOUT:   .Greater = imports.%import_ref.6
// CHECK:STDOUT:   .GreaterOrEquivalent = imports.%import_ref.7
// CHECK:STDOUT:   witness = (imports.%import_ref.8, imports.%import_ref.9, imports.%import_ref.10, imports.%import_ref.11)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %Less.ref: %.5 = name_ref Less, imports.%import_ref.4 [template = constants.%.6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Less(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Less.%Self (%Self)](%other: @Less.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %LessOrEquivalent.ref: %.7 = name_ref LessOrEquivalent, imports.%import_ref.5 [template = constants.%.8]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @LessOrEquivalent(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @LessOrEquivalent.%Self (%Self)](%other: @LessOrEquivalent.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %Greater.ref: %.9 = name_ref Greater, imports.%import_ref.6 [template = constants.%.10]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Greater(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Greater.%Self (%Self)](%other: @Greater.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %GreaterOrEquivalent.ref: %.11 = name_ref GreaterOrEquivalent, imports.%import_ref.7 [template = constants.%.12]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GreaterOrEquivalent(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @GreaterOrEquivalent.%Self (%Self)](%other: @GreaterOrEquivalent.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Less(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LessOrEquivalent(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Greater(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GreaterOrEquivalent(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
