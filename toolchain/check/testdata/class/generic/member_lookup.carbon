// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessDerived[T:! type](x: Derived(T)) -> T {
  return x.d;
}

fn AccessBase[T:! type](x: Derived(T)) -> T {
  return x.b;
}

fn AccessConcrete(x: Derived(i32)) -> i32 {
  return x.b;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessMissingBase[T:! type](x: Base(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Base(T)` [MemberNameNotFoundInScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingDerived[T:! type](x: Derived(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Derived(T)` [MemberNameNotFoundInScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingConcrete(x: Derived(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+3]]:10: error: member name `nonesuch` not found in `Derived(i32)` [MemberNameNotFoundInScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  return x.nonesuch;
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [template]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [template]
// CHECK:STDOUT:   %Base.1: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.elem.1: type = unbound_element_type %Base.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.b.1: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.b.1 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [template]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [template]
// CHECK:STDOUT:   %Derived.1: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.elem.1: type = unbound_element_type %Derived.1, %Base.1 [symbolic]
// CHECK:STDOUT:   %Derived.elem.2: type = unbound_element_type %Derived.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.1: type = struct_type {.base: %Base.1, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.d.1 [symbolic]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [template]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [template]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [template]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Derived.2: type = class_type @Derived, @Derived(%i32) [template]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [template]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [template]
// CHECK:STDOUT:   %Base.2: type = class_type @Base, @Base(%i32) [template]
// CHECK:STDOUT:   %Derived.elem.3: type = unbound_element_type %Derived.2, %Base.2 [template]
// CHECK:STDOUT:   %Derived.elem.4: type = unbound_element_type %Derived.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.d.3: type = struct_type {.base: %Base.2, .d: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.d.3 [template]
// CHECK:STDOUT:   %Base.elem.2: type = unbound_element_type %Base.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.b.2: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %struct_type.b.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived.decl
// CHECK:STDOUT:     .AccessBase = %AccessBase.decl
// CHECK:STDOUT:     .AccessConcrete = %AccessConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [template = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt.loc4_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_17.1, runtime_param<invalid> [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [template = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_15.1, runtime_param<invalid> [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [template = constants.%AccessDerived] {
// CHECK:STDOUT:     %T.patt.loc13_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_18.1, runtime_param<invalid> [symbolic = %T.patt.loc13_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @AccessDerived.%T.loc13_18.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessDerived.%T.loc13_18.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [template = constants.%Derived.generic]
// CHECK:STDOUT:     %T.ref.loc13_39: type = name_ref T, %T.loc13_18.1 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:     %Derived.loc13_40.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc13_40.2 (constants.%Derived.1)]
// CHECK:STDOUT:     %T.ref.loc13_46: type = name_ref T, %T.loc13_18.1 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_18.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessDerived.%Derived.loc13_40.2 (%Derived.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @AccessDerived.%Derived.loc13_40.2 (%Derived.1) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessDerived.%T.loc13_18.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @AccessDerived.%T.loc13_18.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [template = constants.%AccessBase] {
// CHECK:STDOUT:     %T.patt.loc17_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc17_15.1, runtime_param<invalid> [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessBase.%Derived.loc17_37.2 (%Derived.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessBase.%Derived.loc17_37.2 (%Derived.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @AccessBase.%T.loc17_15.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessBase.%T.loc17_15.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [template = constants.%Derived.generic]
// CHECK:STDOUT:     %T.ref.loc17_36: type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %Derived.loc17_37.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc17_37.2 (constants.%Derived.1)]
// CHECK:STDOUT:     %T.ref.loc17_43: type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc17_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessBase.%Derived.loc17_37.2 (%Derived.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @AccessBase.%Derived.loc17_37.2 (%Derived.1) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessBase.%T.loc17_15.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @AccessBase.%T.loc17_15.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [template = constants.%AccessConcrete] {
// CHECK:STDOUT:     %x.patt: %Derived.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Derived.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [template = constants.%Derived.generic]
// CHECK:STDOUT:     %int_32.loc21_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_30: init type = call constants.%Int(%int_32.loc21_30) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_33.1: type = value_of_initializer %int.make_type_signed.loc21_30 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_33.2: type = converted %int.make_type_signed.loc21_30, %.loc21_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %Derived: type = class_type @Derived, @Derived(constants.%i32) [template = constants.%Derived.2]
// CHECK:STDOUT:     %int_32.loc21_39: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_39: init type = call constants.%Int(%int_32.loc21_39) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_39.1: type = value_of_initializer %int.make_type_signed.loc21_39 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_39.2: type = converted %int.make_type_signed.loc21_39, %.loc21_39.1 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %Derived.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %Derived.2 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.1)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.1), @Base.%T.loc4_17.2 (%T) [symbolic = %Base.elem (constants.%Base.elem.1)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.1)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.b (%struct_type.b.1) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.1 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.1) = field_decl b, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.b.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.1
// CHECK:STDOUT:     .b = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.2) [symbolic = %Base.loc9_22.2 (constants.%Base.1)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.2) [symbolic = %Derived (constants.%Derived.1)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type @Derived.%Derived (%Derived.1), @Derived.%Base.loc9_22.2 (%Base.1) [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.1)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type @Derived.%Derived (%Derived.1), @Derived.%T.loc8_15.2 (%T) [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.1), .d: @Derived.%T.loc8_15.2 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.1)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @Derived.%struct_type.base.d (%struct_type.base.d.1) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [template = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.1)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.1) = base_decl %Base.loc9_22.1, element0 [template]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10: @Derived.%Derived.elem.loc10 (%Derived.elem.2) = field_decl d, element1 [template]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.d.1 [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.1
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_18.1: type) {
// CHECK:STDOUT:   %T.loc13_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc13_40.2: type = class_type @Derived, @Derived(%T.loc13_18.2) [symbolic = %Derived.loc13_40.2 (constants.%Derived.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type @AccessDerived.%Derived.loc13_40.2 (%Derived.1), @AccessDerived.%T.loc13_18.2 (%T) [symbolic = %Derived.elem (constants.%Derived.elem.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.1)) -> @AccessDerived.%T.loc13_18.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_40.2 (%Derived.1) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.2) = name_ref d, @Derived.%.loc10 [template = @Derived.%.loc10]
// CHECK:STDOUT:     %.loc14_11.1: ref @AccessDerived.%T.loc13_18.2 (%T) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc14_11.2: @AccessDerived.%T.loc13_18.2 (%T) = bind_value %.loc14_11.1
// CHECK:STDOUT:     return %.loc14_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc17_15.1: type) {
// CHECK:STDOUT:   %T.loc17_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc17_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc17_37.2: type = class_type @Derived, @Derived(%T.loc17_15.2) [symbolic = %Derived.loc17_37.2 (constants.%Derived.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc17_15.2) [symbolic = %Base (constants.%Base.1)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @AccessBase.%Base (%Base.1), @AccessBase.%T.loc17_15.2 (%T) [symbolic = %Base.elem (constants.%Base.elem.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @AccessBase.%Derived.loc17_37.2 (%Derived.1)) -> @AccessBase.%T.loc17_15.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc17_37.2 (%Derived.1) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.1) = name_ref b, @Base.%.loc5 [template = @Base.%.loc5]
// CHECK:STDOUT:     %.loc18_11.1: ref @AccessBase.%Base (%Base.1) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc18_11.2: ref @AccessBase.%Base (%Base.1) = converted %x.ref, %.loc18_11.1
// CHECK:STDOUT:     %.loc18_11.3: ref @AccessBase.%T.loc17_15.2 (%T) = class_element_access %.loc18_11.2, element0
// CHECK:STDOUT:     %.loc18_11.4: @AccessBase.%T.loc17_15.2 (%T) = bind_value %.loc18_11.3
// CHECK:STDOUT:     return %.loc18_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessConcrete(%x.param_patt: %Derived.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.2 = name_ref x, %x
// CHECK:STDOUT:   %b.ref: %Base.elem.2 = name_ref b, @Base.%.loc5 [template = @Base.%.loc5]
// CHECK:STDOUT:   %.loc22_11.1: ref %Base.2 = class_element_access %x.ref, element0
// CHECK:STDOUT:   %.loc22_11.2: ref %Base.2 = converted %x.ref, %.loc22_11.1
// CHECK:STDOUT:   %.loc22_11.3: ref %i32 = class_element_access %.loc22_11.2, element0
// CHECK:STDOUT:   %.loc22_11.4: %i32 = bind_value %.loc22_11.3
// CHECK:STDOUT:   return %.loc22_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base => constants.%Base.1
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.1
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.1
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc4_17.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.1
// CHECK:STDOUT:   %Derived => constants.%Derived.1
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.1
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.2
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.1
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@Derived.%T.loc8_15.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(%T.loc8_15.2) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessDerived.%T.loc13_18.2) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc13_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_18.2 => constants.%T
// CHECK:STDOUT:   %Derived.loc13_40.2 => constants.%Derived.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessBase.%T.loc17_15.2) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T) {
// CHECK:STDOUT:   %T.loc17_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc17_15.2 => constants.%T
// CHECK:STDOUT:   %Derived.loc17_37.2 => constants.%Derived.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@AccessBase.%T.loc17_15.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.2
// CHECK:STDOUT:   %Derived => constants.%Derived.2
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.3
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.4
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.3
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base => constants.%Base.2
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.2
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.2
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [template]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [template]
// CHECK:STDOUT:   %Base.1: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.elem.1: type = unbound_element_type %Base.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.b.1: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.b.1 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [template]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [template]
// CHECK:STDOUT:   %Derived.1: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.elem.1: type = unbound_element_type %Derived.1, %Base.1 [symbolic]
// CHECK:STDOUT:   %Derived.elem.2: type = unbound_element_type %Derived.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.1: type = struct_type {.base: %Base.1, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.d.1 [symbolic]
// CHECK:STDOUT:   %AccessMissingBase.type: type = fn_type @AccessMissingBase [template]
// CHECK:STDOUT:   %AccessMissingBase: %AccessMissingBase.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMissingDerived.type: type = fn_type @AccessMissingDerived [template]
// CHECK:STDOUT:   %AccessMissingDerived: %AccessMissingDerived.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Derived.2: type = class_type @Derived, @Derived(%i32) [template]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [template]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [template]
// CHECK:STDOUT:   %Base.2: type = class_type @Base, @Base(%i32) [template]
// CHECK:STDOUT:   %Derived.elem.3: type = unbound_element_type %Derived.2, %Base.2 [template]
// CHECK:STDOUT:   %Derived.elem.4: type = unbound_element_type %Derived.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.d.3: type = struct_type {.base: %Base.2, .d: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.d.3 [template]
// CHECK:STDOUT:   %Base.elem.2: type = unbound_element_type %Base.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.b.2: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %struct_type.b.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessMissingBase = %AccessMissingBase.decl
// CHECK:STDOUT:     .AccessMissingDerived = %AccessMissingDerived.decl
// CHECK:STDOUT:     .AccessMissingConcrete = %AccessMissingConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [template = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt.loc4_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_17.1, runtime_param<invalid> [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [template = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_15.1, runtime_param<invalid> [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingBase.decl: %AccessMissingBase.type = fn_decl @AccessMissingBase [template = constants.%AccessMissingBase] {
// CHECK:STDOUT:     %T.patt.loc13_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_22.1, runtime_param<invalid> [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @AccessMissingBase.%T.loc13_22.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingBase.%T.loc13_22.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [template = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref.loc13_40: type = name_ref T, %T.loc13_22.1 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc13_41.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc13_41.2 (constants.%Base.1)]
// CHECK:STDOUT:     %T.ref.loc13_47: type = name_ref T, %T.loc13_22.1 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingBase.%Base.loc13_41.2 (%Base.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @AccessMissingBase.%Base.loc13_41.2 (%Base.1) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingBase.%T.loc13_22.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @AccessMissingBase.%T.loc13_22.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingDerived.decl: %AccessMissingDerived.type = fn_decl @AccessMissingDerived [template = constants.%AccessMissingDerived] {
// CHECK:STDOUT:     %T.patt.loc21_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc21_25.1, runtime_param<invalid> [symbolic = %T.patt.loc21_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @AccessMissingDerived.%T.loc21_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingDerived.%T.loc21_25.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [template = constants.%Derived.generic]
// CHECK:STDOUT:     %T.ref.loc21_46: type = name_ref T, %T.loc21_25.1 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:     %Derived.loc21_47.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc21_47.2 (constants.%Derived.1)]
// CHECK:STDOUT:     %T.ref.loc21_53: type = name_ref T, %T.loc21_25.1 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc21_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingDerived.%T.loc21_25.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @AccessMissingDerived.%T.loc21_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [template = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %x.patt: %Derived.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Derived.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [template = constants.%Derived.generic]
// CHECK:STDOUT:     %int_32.loc29_37: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc29_37: init type = call constants.%Int(%int_32.loc29_37) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_40.1: type = value_of_initializer %int.make_type_signed.loc29_37 [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_40.2: type = converted %int.make_type_signed.loc29_37, %.loc29_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %Derived: type = class_type @Derived, @Derived(constants.%i32) [template = constants.%Derived.2]
// CHECK:STDOUT:     %int_32.loc29_46: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc29_46: init type = call constants.%Int(%int_32.loc29_46) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_46.1: type = value_of_initializer %int.make_type_signed.loc29_46 [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_46.2: type = converted %int.make_type_signed.loc29_46, %.loc29_46.1 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %Derived.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %Derived.2 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.1)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.1), @Base.%T.loc4_17.2 (%T) [symbolic = %Base.elem (constants.%Base.elem.1)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.1)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.b (%struct_type.b.1) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.1 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.1) = field_decl b, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.b.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.1
// CHECK:STDOUT:     .b = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.2) [symbolic = %Base.loc9_22.2 (constants.%Base.1)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.2) [symbolic = %Derived (constants.%Derived.1)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type @Derived.%Derived (%Derived.1), @Derived.%Base.loc9_22.2 (%Base.1) [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.1)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type @Derived.%Derived (%Derived.1), @Derived.%T.loc8_15.2 (%T) [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.1), .d: @Derived.%T.loc8_15.2 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.1)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @Derived.%struct_type.base.d (%struct_type.base.d.1) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [template = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.1)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.1) = base_decl %Base.loc9_22.1, element0 [template]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10: @Derived.%Derived.elem.loc10 (%Derived.elem.2) = field_decl d, element1 [template]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.d.1 [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.1
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingBase(%T.loc13_22.1: type) {
// CHECK:STDOUT:   %T.loc13_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Base.loc13_41.2: type = class_type @Base, @Base(%T.loc13_22.2) [symbolic = %Base.loc13_41.2 (constants.%Base.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.1)) -> @AccessMissingBase.%T.loc13_22.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingBase.%Base.loc13_41.2 (%Base.1) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [template = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingDerived(%T.loc21_25.1: type) {
// CHECK:STDOUT:   %T.loc21_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc21_47.2: type = class_type @Derived, @Derived(%T.loc21_25.2) [symbolic = %Derived.loc21_47.2 (constants.%Derived.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1)) -> @AccessMissingDerived.%T.loc21_25.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.1) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [template = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMissingConcrete(%x.param_patt: %Derived.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.2 = name_ref x, %x
// CHECK:STDOUT:   %nonesuch.ref: <error> = name_ref nonesuch, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base => constants.%Base.1
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.1
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.1
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc4_17.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.1
// CHECK:STDOUT:   %Derived => constants.%Derived.1
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.1
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.2
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.1
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@Derived.%T.loc8_15.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(%T.loc8_15.2) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@AccessMissingBase.%T.loc13_22.2) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingBase(constants.%T) {
// CHECK:STDOUT:   %T.loc13_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_22.2 => constants.%T
// CHECK:STDOUT:   %Base.loc13_41.2 => constants.%Base.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessMissingDerived.%T.loc21_25.2) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc21_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_25.2 => constants.%T
// CHECK:STDOUT:   %Derived.loc21_47.2 => constants.%Derived.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.2
// CHECK:STDOUT:   %Derived => constants.%Derived.2
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.3
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.4
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.3
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base => constants.%Base.2
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.2
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.2
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
