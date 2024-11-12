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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Eq` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestNotEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Eq` in type `D` that does not implement that interface [MissingImplInMemberAccess]
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
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+10]]:15: error: cannot implicitly convert from `D` to `C` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:               ^
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+7]]:15: note: type `D` does not implement interface `ImplicitAs(C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:               ^
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE-11]]:21: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Equal[self: C](other: C) -> bool;
  // CHECK:STDERR:                     ^~~~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestLhsBad(a: D, b: C) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: error: cannot access member of interface `Eq` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  return a != b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Eq.type: type = facet_type <@Eq> [template]
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %Equal.1: %Equal.type.1 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %NotEqual.1: %NotEqual.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %Equal.2: %Equal.type.2 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %NotEqual.2: %NotEqual.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%Equal.1, %NotEqual.1) [template]
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Eq.type, %Equal.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %Eq.type, %NotEqual.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element1, imports.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Eq = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%Eq.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.3: %.5 = import_ref Core//prelude/operators/comparison, inst+40, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4: %.7 = import_ref Core//prelude/operators/comparison, inst+68, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.5: %Equal.type.2 = import_ref Core//prelude/operators/comparison, inst+35, loaded [template = constants.%Equal.2]
// CHECK:STDOUT:   %import_ref.6: %NotEqual.type.2 = import_ref Core//prelude/operators/comparison, inst+63, loaded [template = constants.%NotEqual.2]
// CHECK:STDOUT:   %import_ref.7: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/comparison, inst+63, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, imports.%import_ref.1 [template = constants.%Eq.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [template = constants.%TestEqual] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc11_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc11_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %bool.make_type, %.loc11_29.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [template = constants.%TestNotEqual] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc15_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %bool.make_type, %.loc15_32.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
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
// CHECK:STDOUT: impl @impl: %C.ref as %Eq.ref {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.1 = fn_decl @Equal.1 [template = constants.%Equal.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc7_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc7_34.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc7_34.2: type = converted %bool.make_type, %.loc7_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.1 = fn_decl @NotEqual.1 [template = constants.%NotEqual.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc8_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_37.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %bool.make_type, %.loc8_37.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
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
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal.2(constants.%Self: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Equal.2.%Self (%Self)](%other.param_patt: @Equal.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual.2(constants.%Self: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @NotEqual.2.%Self (%Self)](%other.param_patt: @NotEqual.2.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc12_12.1: %Equal.type.2 = interface_witness_access constants.%.3, element0 [template = constants.%Equal.1]
// CHECK:STDOUT:   %.loc12_12.2: <bound method> = bound_method %a.ref, %.loc12_12.1
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc12_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Equal.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Equal.call, %.loc12_16.1
// CHECK:STDOUT:   return %.loc12_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a.param_patt: %C, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc16_12.1: %NotEqual.type.2 = interface_witness_access constants.%.3, element1 [template = constants.%NotEqual.1]
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
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %Eq.type: type = facet_type <@Eq> [template]
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Eq.type, %Equal.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [template]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %Eq.type, %NotEqual.type [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element1, imports.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Eq = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%Eq.type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.4: %.4 = import_ref Core//prelude/operators/comparison, inst+40, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.5: %.6 = import_ref Core//prelude/operators/comparison, inst+68, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/comparison, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/comparison, inst+63, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/comparison, inst+63, unloaded
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
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [template = constants.%TestEqual] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc6_17: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc6_23: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %bool.make_type, %.loc6_29.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [template = constants.%TestNotEqual] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc14_20: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %D.ref.loc14_26: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_32.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc14_32.2: type = converted %bool.make_type, %.loc14_32.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
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
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(constants.%Self: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Equal.%Self (%Self)](%other.param_patt: @Equal.%Self (%Self)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a.param_patt: %D, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual(constants.%Self: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @NotEqual.%Self (%Self)](%other.param_patt: @NotEqual.%Self (%Self)) -> bool;
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
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %Eq.type: type = facet_type <@Eq> [template]
// CHECK:STDOUT:   %Self.1: %Eq.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %Equal.1: %Equal.type.1 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %NotEqual.1: %NotEqual.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Equal.type.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %Equal.2: %Equal.type.2 = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.type.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %NotEqual.2: %NotEqual.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%Equal.1, %NotEqual.1) [template]
// CHECK:STDOUT:   %TestRhsBad.type: type = fn_type @TestRhsBad [template]
// CHECK:STDOUT:   %TestRhsBad: %TestRhsBad.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Eq.type, %Equal.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.8 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.3: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%C)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.13 [template]
// CHECK:STDOUT:   %.11: %.7 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %TestLhsBad.type: type = fn_type @TestLhsBad [template]
// CHECK:STDOUT:   %TestLhsBad: %TestLhsBad.type = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %Eq.type, %NotEqual.type.2 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element1, imports.%import_ref.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Eq = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.7
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/comparison, inst+3, loaded [template = constants.%Eq.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/comparison, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.3: %.5 = import_ref Core//prelude/operators/comparison, inst+40, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4: %.12 = import_ref Core//prelude/operators/comparison, inst+68, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.5: %Equal.type.2 = import_ref Core//prelude/operators/comparison, inst+35, loaded [template = constants.%Equal.2]
// CHECK:STDOUT:   %import_ref.6: %NotEqual.type.2 = import_ref Core//prelude/operators/comparison, inst+63, loaded [template = constants.%NotEqual.2]
// CHECK:STDOUT:   %import_ref.7: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/comparison, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.9: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+48, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+54, unloaded
// CHECK:STDOUT:   %import_ref.11: @ImplicitAs.%.1 (%.7) = import_ref Core//prelude/operators/as, inst+76, loaded [symbolic = @ImplicitAs.%.2 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/comparison, inst+63, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, imports.%import_ref.1 [template = constants.%Eq.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRhsBad.decl: %TestRhsBad.type = fn_decl @TestRhsBad [template = constants.%TestRhsBad] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc12_30.2: type = converted %bool.make_type, %.loc12_30.1 [template = bool]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLhsBad.decl: %TestLhsBad.type = fn_decl @TestLhsBad [template = constants.%TestLhsBad] {
// CHECK:STDOUT:     %a.patt: %D = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %D = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc26_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc26_30.2: type = converted %bool.make_type, %.loc26_30.1 [template = bool]
// CHECK:STDOUT:     %a.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
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
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.7) = assoc_entity element0, imports.%import_ref.13 [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.10
// CHECK:STDOUT:     .Convert = imports.%import_ref.11
// CHECK:STDOUT:     witness = (imports.%import_ref.12)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %Eq.ref {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.1 = fn_decl @Equal.1 [template = constants.%Equal.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_34.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %bool.make_type, %.loc8_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.1 = fn_decl @NotEqual.1 [template = constants.%NotEqual.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_37.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc9_37.2: type = converted %bool.make_type, %.loc9_37.1 [template = bool]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
// CHECK:STDOUT:   witness = %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[%self.param_patt: %C](%other.param_patt: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal.2(constants.%Self.1: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Equal.2.%Self (%Self.1)](%other.param_patt: @Equal.2.%Self (%Self.1)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NotEqual.2(constants.%Self.1: %Eq.type) {
// CHECK:STDOUT:   %Self: %Eq.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @NotEqual.2.%Self (%Self.1)](%other.param_patt: @NotEqual.2.%Self (%Self.1)) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a.param_patt: %C, %b.param_patt: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %.loc23_12.1: %Equal.type.2 = interface_witness_access constants.%.3, element0 [template = constants.%Equal.1]
// CHECK:STDOUT:   %.loc23_12.2: <bound method> = bound_method %a.ref, %.loc23_12.1
// CHECK:STDOUT:   %.loc23_15: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc23_12.2(%a.ref, <error>)
// CHECK:STDOUT:   %.loc23_16.1: bool = value_of_initializer %Equal.call
// CHECK:STDOUT:   %.loc23_16.2: bool = converted %Equal.call, %.loc23_16.1
// CHECK:STDOUT:   return %.loc23_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a.param_patt: %D, %b.param_patt: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
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
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C) {
// CHECK:STDOUT:   %Dest => constants.%C
// CHECK:STDOUT:   %Dest.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
