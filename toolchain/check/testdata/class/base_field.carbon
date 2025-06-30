// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_field.carbon

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

fn Access(p: Derived*) -> i32* {
  return &(*p).c;
}

// CHECK:STDOUT: --- base_field.carbon
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
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fe8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fe8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc28_30: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc28: type = splice_block %ptr.loc28_21 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc28_21: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc18: %Base.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.c [concrete = constants.%complete_type.ebc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .b = %.loc17
// CHECK:STDOUT:   .c = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc22: %Derived.elem.69e = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc24: %Derived.elem.344 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc25: %Derived.elem.344 = field_decl e, element2 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete = constants.%struct_type.base.d.e.6a7]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d.e [concrete = constants.%complete_type.401]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   .d = %.loc24
// CHECK:STDOUT:   .e = %.loc25
// CHECK:STDOUT:   .c = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p.param: %ptr.404) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %.loc29_12: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: %Base.elem = name_ref c, @Base.%.loc18 [concrete = @Base.%.loc18]
// CHECK:STDOUT:   %.loc29_15.1: ref %Base = class_element_access %.loc29_12, element0
// CHECK:STDOUT:   %.loc29_15.2: ref %Base = converted %.loc29_12, %.loc29_15.1
// CHECK:STDOUT:   %.loc29_15.3: ref %i32 = class_element_access %.loc29_15.2, element2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc29_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
