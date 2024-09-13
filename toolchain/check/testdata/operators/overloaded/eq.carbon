// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.Eq {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestEqual(a: C, b: C) -> bool {
  return a == b;
}

fn TestNotEqual(a: C, b: C) -> bool {
  return a != b;
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface `Eq` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestNotEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface `Eq` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a != b;
}

// --- fail_no_impl_for_args.carbon

package FailNoImplForArgs;

class C {};
class D {};

impl C as Core.Eq {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestRhsBad(a: C, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+10]]:10: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+7]]:10: Type `D` does not implement interface `ImplicitAs`.
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE-11]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Equal[self: C](other: C) -> bool;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestLhsBad(a: D, b: C) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface `Eq` in type `D` that does not implement that interface.
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  return a != b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %Equal.1: %Equal.type.1 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %NotEqual.1: %NotEqual.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %Equal.2: %Equal.type.2 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %NotEqual.2: %NotEqual.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Equal.1, %NotEqual.1) [template]
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %Equal.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.2, %NotEqual.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element1, imports.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Eq = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.3: %.6 = import_ref Core//prelude/operators/comparison, inst+31, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: %.8 = import_ref Core//prelude/operators/comparison, inst+52, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.5: %Equal.type.2 = import_ref Core//prelude/operators/comparison, inst+26, loaded [template = constants.%Equal.2]
// CHECK:STDOUT:   %import_ref.6: %NotEqual.type.2 = import_ref Core//prelude/operators/comparison, inst+47, loaded [template = constants.%NotEqual.2]
// CHECK:STDOUT:   %import_ref.7: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/comparison, inst+47, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [template = constants.%TestEqual] {
// CHECK:STDOUT:     %C.ref.loc11_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc11_14.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestEqual.%a: %C = bind_name a, %a.loc11_14.1
// CHECK:STDOUT:     %C.ref.loc11_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc11_20.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestEqual.%b: %C = bind_name b, %b.loc11_20.1
// CHECK:STDOUT:     %bool.make_type.loc11: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %bool.make_type.loc11, %.loc11_29.1 [template = bool]
// CHECK:STDOUT:     @TestEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [template = constants.%TestNotEqual] {
// CHECK:STDOUT:     %C.ref.loc15_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_17.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestNotEqual.%a: %C = bind_name a, %a.loc15_17.1
// CHECK:STDOUT:     %C.ref.loc15_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc15_23.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestNotEqual.%b: %C = bind_name b, %b.loc15_23.1
// CHECK:STDOUT:     %bool.make_type.loc15: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %bool.make_type.loc15 [template = bool]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %bool.make_type.loc15, %.loc15_32.1 [template = bool]
// CHECK:STDOUT:     @TestNotEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Equal = imports.%import_ref.3
// CHECK:STDOUT:   .NotEqual = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5, imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.2 {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.1 = fn_decl @Equal.1 [template = constants.%Equal.1] {
// CHECK:STDOUT:     %C.ref.loc7_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc7_12.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc7_12.2: %C = bind_name self, %self.loc7_12.1
// CHECK:STDOUT:     %C.ref.loc7_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc7_21.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc7_21.2: %C = bind_name other, %other.loc7_21.1
// CHECK:STDOUT:     %bool.make_type.loc7: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc7_34.1: type = value_of_initializer %bool.make_type.loc7 [template = bool]
// CHECK:STDOUT:     %.loc7_34.2: type = converted %bool.make_type.loc7, %.loc7_34.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc7: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.1 = fn_decl @NotEqual.1 [template = constants.%NotEqual.1] {
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_15.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc8_15.2: %C = bind_name self, %self.loc8_15.1
// CHECK:STDOUT:     %C.ref.loc8_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc8_24.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc8_24.2: %C = bind_name other, %other.loc8_24.1
// CHECK:STDOUT:     %bool.make_type.loc8: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_37.1: type = value_of_initializer %bool.make_type.loc8 [template = bool]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %bool.make_type.loc8, %.loc8_37.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
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
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc7_12.2: %C](@impl.%other.loc7_21.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc8_15.2: %C](@impl.%other.loc8_24.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Equal.2.%Self (%Self)](%other: @Equal.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @NotEqual.2.%Self (%Self)](%other: @NotEqual.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Equal.ref: %.6 = name_ref Equal, imports.%import_ref.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_12.1: %Equal.type.2 = interface_witness_access @impl.%.loc6, element0 [template = constants.%Equal.1]
// CHECK:STDOUT:   %.loc12_12.2: <bound method> = bound_method %a.ref, %.loc12_12.1
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc12_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Equal.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Equal.call, %.loc12_16.1
// CHECK:STDOUT:   return %.loc12_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: %C, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %NotEqual.ref: %.8 = name_ref NotEqual, imports.%import_ref.4 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_12.1: %NotEqual.type.2 = interface_witness_access @impl.%.loc6, element1 [template = constants.%NotEqual.1]
// CHECK:STDOUT:   %.loc16_12.2: <bound method> = bound_method %a.ref, %.loc16_12.1
// CHECK:STDOUT:   %NotEqual.call: init bool = call %.loc16_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_16.1: bool = value_of_initializer %NotEqual.call
// CHECK:STDOUT:   %.loc16_16.2: bool = converted %NotEqual.call, %.loc16_16.1
// CHECK:STDOUT:   return %.loc16_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotEqual.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotEqual.2(constants.%C) {
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
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Equal.type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [template]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %NotEqual.type [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element1, imports.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Eq = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.4: %.5 = import_ref Core//prelude/operators/comparison, inst+31, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.5: %.7 = import_ref Core//prelude/operators/comparison, inst+52, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/comparison, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/comparison, inst+47, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/comparison, inst+47, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [template = constants.%TestEqual] {
// CHECK:STDOUT:     %D.ref.loc6_17: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc6_14.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestEqual.%a: %D = bind_name a, %a.loc6_14.1
// CHECK:STDOUT:     %D.ref.loc6_23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc6_20.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestEqual.%b: %D = bind_name b, %b.loc6_20.1
// CHECK:STDOUT:     %bool.make_type.loc6: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %bool.make_type.loc6 [template = bool]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %bool.make_type.loc6, %.loc6_29.1 [template = bool]
// CHECK:STDOUT:     @TestEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [template = constants.%TestNotEqual] {
// CHECK:STDOUT:     %D.ref.loc14_20: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc14_17.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestNotEqual.%a: %D = bind_name a, %a.loc14_17.1
// CHECK:STDOUT:     %D.ref.loc14_26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc14_23.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestNotEqual.%b: %D = bind_name b, %b.loc14_23.1
// CHECK:STDOUT:     %bool.make_type.loc14: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_32.1: type = value_of_initializer %bool.make_type.loc14 [template = bool]
// CHECK:STDOUT:     %.loc14_32.2: type = converted %bool.make_type.loc14, %.loc14_32.1 [template = bool]
// CHECK:STDOUT:     @TestNotEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Equal = imports.%import_ref.4
// CHECK:STDOUT:   .NotEqual = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6, imports.%import_ref.7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %Equal.ref: %.5 = name_ref Equal, imports.%import_ref.4 [template = constants.%.6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Equal.%Self (%Self)](%other: @Equal.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: %D, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %NotEqual.ref: %.7 = name_ref NotEqual, imports.%import_ref.5 [template = constants.%.8]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @NotEqual.%Self (%Self)](%other: @NotEqual.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotEqual(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl_for_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %Equal.1: %Equal.type.1 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %NotEqual.1: %NotEqual.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %Equal.2: %Equal.type.2 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %NotEqual.2: %NotEqual.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Equal.1, %NotEqual.1) [template]
// CHECK:STDOUT:   %TestRhsBad.type: type = fn_type @TestRhsBad [template]
// CHECK:STDOUT:   %TestRhsBad: %TestRhsBad.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %Equal.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.8: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%.1 (%.8) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %.8 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %.11: type = interface_type @ImplicitAs, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %.11, %Convert.type.2 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.13 [template]
// CHECK:STDOUT:   %.14: %.9 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %TestLhsBad.type: type = fn_type @TestLhsBad [template]
// CHECK:STDOUT:   %TestLhsBad: %TestLhsBad.type = struct_value () [template]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %.2, %NotEqual.type.2 [template]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element1, imports.%import_ref.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Eq = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.7
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.3: %.6 = import_ref Core//prelude/operators/comparison, inst+31, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: %.15 = import_ref Core//prelude/operators/comparison, inst+52, loaded [template = constants.%.16]
// CHECK:STDOUT:   %import_ref.5: %Equal.type.2 = import_ref Core//prelude/operators/comparison, inst+26, loaded [template = constants.%Equal.2]
// CHECK:STDOUT:   %import_ref.6: %NotEqual.type.2 = import_ref Core//prelude/operators/comparison, inst+47, loaded [template = constants.%NotEqual.2]
// CHECK:STDOUT:   %import_ref.7: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.9: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.11: @ImplicitAs.%.2 (%.9) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/comparison, inst+47, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestRhsBad = %TestRhsBad.decl
// CHECK:STDOUT:     .TestLhsBad = %TestLhsBad.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRhsBad.decl: %TestRhsBad.type = fn_decl @TestRhsBad [template = constants.%TestRhsBad] {
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc12_15.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @TestRhsBad.%a: %C = bind_name a, %a.loc12_15.1
// CHECK:STDOUT:     %D.ref.loc12: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc12_21.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @TestRhsBad.%b: %D = bind_name b, %b.loc12_21.1
// CHECK:STDOUT:     %bool.make_type.loc12: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12_30.1: type = value_of_initializer %bool.make_type.loc12 [template = bool]
// CHECK:STDOUT:     %.loc12_30.2: type = converted %bool.make_type.loc12, %.loc12_30.1 [template = bool]
// CHECK:STDOUT:     @TestRhsBad.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLhsBad.decl: %TestLhsBad.type = fn_decl @TestLhsBad [template = constants.%TestLhsBad] {
// CHECK:STDOUT:     %D.ref.loc26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc26_15.1: %D = param a, runtime_param0
// CHECK:STDOUT:     @TestLhsBad.%a: %D = bind_name a, %a.loc26_15.1
// CHECK:STDOUT:     %C.ref.loc26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc26_21.1: %C = param b, runtime_param1
// CHECK:STDOUT:     @TestLhsBad.%b: %C = bind_name b, %b.loc26_21.1
// CHECK:STDOUT:     %bool.make_type.loc26: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc26_30.1: type = value_of_initializer %bool.make_type.loc26 [template = bool]
// CHECK:STDOUT:     %.loc26_30.2: type = converted %bool.make_type.loc26, %.loc26_30.1 [template = bool]
// CHECK:STDOUT:     @TestLhsBad.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Equal = imports.%import_ref.3
// CHECK:STDOUT:   .NotEqual = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5, imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.8), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.9) = assoc_entity element0, imports.%import_ref.13 [symbolic = %.3 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.10
// CHECK:STDOUT:     .Convert = imports.%import_ref.11
// CHECK:STDOUT:     witness = (imports.%import_ref.12)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.2 {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.1 = fn_decl @Equal.1 [template = constants.%Equal.1] {
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_12.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc8_12.2: %C = bind_name self, %self.loc8_12.1
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc8_21.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc8_21.2: %C = bind_name other, %other.loc8_21.1
// CHECK:STDOUT:     %bool.make_type.loc8: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_34.1: type = value_of_initializer %bool.make_type.loc8 [template = bool]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %bool.make_type.loc8, %.loc8_34.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.1 = fn_decl @NotEqual.1 [template = constants.%NotEqual.1] {
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_15.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc9_15.2: %C = bind_name self, %self.loc9_15.1
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_24.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc9_24.2: %C = bind_name other, %other.loc9_24.1
// CHECK:STDOUT:     %bool.make_type.loc9: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_37.1: type = value_of_initializer %bool.make_type.loc9 [template = bool]
// CHECK:STDOUT:     %.loc9_37.2: type = converted %bool.make_type.loc9, %.loc9_37.1 [template = bool]
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
// CHECK:STDOUT:   witness = %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc8_12.2: %C](@impl.%other.loc8_21.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc9_15.2: %C](@impl.%other.loc9_24.2: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal.2(constants.%Self.1: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Equal.2.%Self (%Self.1)](%other: @Equal.2.%Self (%Self.1)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual.2(constants.%Self.1: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @NotEqual.2.%Self (%Self.1)](%other: @NotEqual.2.%Self (%Self.1)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a: %C, %b: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %Equal.ref: %.6 = name_ref Equal, imports.%import_ref.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc23_12.1: %Equal.type.2 = interface_witness_access @impl.%.loc7, element0 [template = constants.%Equal.1]
// CHECK:STDOUT:   %.loc23_12.2: <bound method> = bound_method %a.ref, %.loc23_12.1
// CHECK:STDOUT:   %.loc23_12.3: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C) [template = constants.%.11]
// CHECK:STDOUT:   %.loc23_12.4: %.12 = specific_constant imports.%import_ref.11, @ImplicitAs(constants.%C) [template = constants.%.13]
// CHECK:STDOUT:   %Convert.ref: %.12 = name_ref Convert, %.loc23_12.4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc23_12.5: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc23_12.2(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc23_16.1: bool = value_of_initializer %Equal.call [template = <error>]
// CHECK:STDOUT:   %.loc23_16.2: bool = converted %Equal.call, %.loc23_16.1 [template = <error>]
// CHECK:STDOUT:   return %.loc23_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%.1 (%.8)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a: %D, %b: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %NotEqual.ref: %.15 = name_ref NotEqual, imports.%import_ref.4 [template = constants.%.16]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal.2(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotEqual.2(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotEqual.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C) {
// CHECK:STDOUT:   %Dest => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT:   %.3 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
