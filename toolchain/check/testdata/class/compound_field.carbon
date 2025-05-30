// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ebc: <witness> = complete_type_witness %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.69e: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.344: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e.6a7: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.401: <witness> = complete_type_witness %struct_type.base.d.e.6a7 [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref.loc27: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fe8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fe8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc35_45: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc35: type = splice_block %ptr.loc35_36 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref.loc35: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc35_36: type = ptr_type %Derived.ref.loc35 [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBaseIndirect.decl: %AccessBaseIndirect.type = fn_decl @AccessBaseIndirect [concrete = constants.%AccessBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fe8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fe8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc39_42: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc39: type = splice_block %ptr.loc39_33 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc39_33: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Base.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.c [concrete = constants.%complete_type.ebc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc15
// CHECK:STDOUT:   .b = %.loc16
// CHECK:STDOUT:   .c = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc21: %Derived.elem.69e = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc23: %Derived.elem.344 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc24: %Derived.elem.344 = field_decl e, element2 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete = constants.%struct_type.base.d.e.6a7]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d.e [concrete = constants.%complete_type.401]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .d = %.loc23
// CHECK:STDOUT:   .e = %.loc24
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc28_10: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref.loc28: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc28_20: %Derived.elem.344 = name_ref d, @Derived.%.loc23 [concrete = @Derived.%.loc23]
// CHECK:STDOUT:   %.loc28_11.1: ref %i32 = class_element_access %d.ref.loc28_10, element1
// CHECK:STDOUT:   %.loc28_11.2: %i32 = bind_value %.loc28_11.1
// CHECK:STDOUT:   return %.loc28_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc32_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc32_11.2: ref %Base = converted %d.ref, %.loc32_11.1
// CHECK:STDOUT:   %.loc32_11.3: ref %i32 = class_element_access %.loc32_11.2, element1
// CHECK:STDOUT:   %.loc32_11.4: %i32 = bind_value %.loc32_11.3
// CHECK:STDOUT:   return %.loc32_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerivedIndirect(%p.param: %ptr.404) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Derived.ref.loc36: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref: %Derived.elem.344 = name_ref d, @Derived.%.loc23 [concrete = @Derived.%.loc23]
// CHECK:STDOUT:   %.loc36_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc36_12.2: ref %i32 = class_element_access %.loc36_12.1, element1
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc36_12.2
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBaseIndirect(%p.param: %ptr.404) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc40_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc40_12.2: ref %Base = class_element_access %.loc40_12.1, element0
// CHECK:STDOUT:   %.loc40_12.3: ref %Base = converted %.loc40_12.1, %.loc40_12.2
// CHECK:STDOUT:   %.loc40_12.4: ref %i32 = class_element_access %.loc40_12.3, element1
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc40_12.4
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
