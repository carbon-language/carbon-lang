// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/compound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/compound_field.carbon

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn AccessDerived(d: Derived) -> i32 {
  return d.(Derived.d);
}

fn AccessBase(d: Derived) -> i32 {
  return d.(Base.b);
}

fn AccessDerivedIndirect(p: Derived*) -> i32* {
  return &p->(Derived.d);
}

fn AccessBaseIndirect(p: Derived*) -> i32* {
  return &p->(Base.b);
}

// CHECK:STDOUT: --- compound_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ebc: <witness> = complete_type_witness %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.69e: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.344: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e.6a7: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.401: <witness> = complete_type_witness %struct_type.base.d.e.6a7 [concrete]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect.type: type = fn_type @AccessDerivedIndirect [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect: %AccessDerivedIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect.type: type = fn_type @AccessBaseIndirect [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect: %AccessBaseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived.decl
// CHECK:STDOUT:     .AccessBase = %AccessBase.decl
// CHECK:STDOUT:     .AccessDerivedIndirect = %AccessDerivedIndirect.decl
// CHECK:STDOUT:     .AccessBaseIndirect = %AccessBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [concrete = constants.%AccessDerived] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref.loc24: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerivedIndirect.decl: %AccessDerivedIndirect.type = fn_decl @AccessDerivedIndirect [concrete = constants.%AccessDerivedIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr.404 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.404 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.235 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.235 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc32_45: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc32: type = splice_block %ptr.loc32_36 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref.loc32: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc32_36: type = ptr_type %Derived.ref.loc32 [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBaseIndirect.decl: %AccessBaseIndirect.type = fn_decl @AccessBaseIndirect [concrete = constants.%AccessBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr.404 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.404 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.235 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.235 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc36_42: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc36: type = splice_block %ptr.loc36_33 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc36_33: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc12_8: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %Base.elem = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc12: ref %Base.elem = var <none>
// CHECK:STDOUT:   %.loc13_8: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc13_3: %Base.elem = var_pattern %.loc13_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc13: ref %Base.elem = var <none>
// CHECK:STDOUT:   %.loc14_8: %Base.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc14_3: %Base.elem = var_pattern %.loc14_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc14: ref %Base.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.c [concrete = constants.%complete_type.ebc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT:   .c = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc18: %Derived.elem.69e = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc20_8: %Derived.elem.344 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc20_3: %Derived.elem.344 = var_pattern %.loc20_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc20: ref %Derived.elem.344 = var <none>
// CHECK:STDOUT:   %.loc21_8: %Derived.elem.344 = field_decl e, element2 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc21_3: %Derived.elem.344 = var_pattern %.loc21_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc21: ref %Derived.elem.344 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d.e.6a7 [concrete = constants.%complete_type.401]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .d = %.loc20_8
// CHECK:STDOUT:   .e = %.loc21_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc25_10: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref.loc25: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc25_20: %Derived.elem.344 = name_ref d, @Derived.%.loc20_8 [concrete = @Derived.%.loc20_8]
// CHECK:STDOUT:   %.loc25_11.1: ref %i32 = class_element_access %d.ref.loc25_10, element1
// CHECK:STDOUT:   %.loc25_11.2: %i32 = bind_value %.loc25_11.1
// CHECK:STDOUT:   return %.loc25_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc13_8 [concrete = @Base.%.loc13_8]
// CHECK:STDOUT:   %.loc29_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc29_11.2: ref %Base = converted %d.ref, %.loc29_11.1
// CHECK:STDOUT:   %.loc29_11.3: ref %i32 = class_element_access %.loc29_11.2, element1
// CHECK:STDOUT:   %.loc29_11.4: %i32 = bind_value %.loc29_11.3
// CHECK:STDOUT:   return %.loc29_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerivedIndirect(%p.param_patt: %ptr.404) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Derived.ref.loc33: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref: %Derived.elem.344 = name_ref d, @Derived.%.loc20_8 [concrete = @Derived.%.loc20_8]
// CHECK:STDOUT:   %.loc33_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc33_12.2: ref %i32 = class_element_access %.loc33_12.1, element1
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc33_12.2
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBaseIndirect(%p.param_patt: %ptr.404) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc13_8 [concrete = @Base.%.loc13_8]
// CHECK:STDOUT:   %.loc37_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc37_12.2: ref %Base = class_element_access %.loc37_12.1, element0
// CHECK:STDOUT:   %.loc37_12.3: ref %Base = converted %.loc37_12.1, %.loc37_12.2
// CHECK:STDOUT:   %.loc37_12.4: ref %i32 = class_element_access %.loc37_12.3, element1
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc37_12.4
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
