// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_derived_to_base.carbon

base class A1 {
  var a: i32;
}

base class A2 {
  var a: i32;
}

class B2 {
  extend base: A2;
  var b: i32;
}

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:38: error: cannot implicitly convert value of type `B2*` to `A1*` [ConversionFailure]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: note: type `B2*` does not implement interface `Core.ImplicitAs(A1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:47: error: cannot implicitly convert value of type `Incomplete*` to `A2*` [ConversionFailure]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:47: note: type `Incomplete*` does not implement interface `Core.ImplicitAs(A2*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR:
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A1.elem: type = unbound_element_type %A1, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %i32 [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %B2.elem.a92: type = unbound_element_type %B2, %A2 [concrete]
// CHECK:STDOUT:   %B2.elem.4b2: type = unbound_element_type %B2, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.b.618: type = struct_type {.base: %A2, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.92f: <witness> = complete_type_witness %struct_type.base.b.618 [concrete]
// CHECK:STDOUT:   %ptr.afe: type = ptr_type %B2 [concrete]
// CHECK:STDOUT:   %ptr.678: type = ptr_type %A1 [concrete]
// CHECK:STDOUT:   %ConvertUnrelated.type: type = fn_type @ConvertUnrelated [concrete]
// CHECK:STDOUT:   %ConvertUnrelated: %ConvertUnrelated.type = struct_value () [concrete]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %ptr.c62: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT:   %ptr.590: type = ptr_type %A2 [concrete]
// CHECK:STDOUT:   %ConvertIncomplete.type: type = fn_type @ConvertIncomplete [concrete]
// CHECK:STDOUT:   %ConvertIncomplete: %ConvertIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .ConvertUnrelated = %ConvertUnrelated.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .ConvertIncomplete = %ConvertIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [concrete = constants.%A1] {} {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [concrete = constants.%A2] {} {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [concrete = constants.%B2] {} {}
// CHECK:STDOUT:   %ConvertUnrelated.decl: %ConvertUnrelated.type = fn_decl @ConvertUnrelated [concrete = constants.%ConvertUnrelated] {
// CHECK:STDOUT:     %p.patt: %ptr.afe = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.afe = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.678 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.678 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A1.ref: type = name_ref A1, file.%A1.decl [concrete = constants.%A1]
// CHECK:STDOUT:     %ptr.loc31_34: type = ptr_type %A1 [concrete = constants.%ptr.678]
// CHECK:STDOUT:     %p.param: %ptr.afe = value_param call_param0
// CHECK:STDOUT:     %.loc31_26: type = splice_block %ptr.loc31_26 [concrete = constants.%ptr.afe] {
// CHECK:STDOUT:       %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:       %ptr.loc31_26: type = ptr_type %B2 [concrete = constants.%ptr.afe]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.afe = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.678 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.678 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %ConvertIncomplete.decl: %ConvertIncomplete.type = fn_decl @ConvertIncomplete [concrete = constants.%ConvertIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.c62 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.c62 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.590 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.590 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A2.ref: type = name_ref A2, file.%A2.decl [concrete = constants.%A2]
// CHECK:STDOUT:     %ptr.loc42_43: type = ptr_type %A2 [concrete = constants.%ptr.590]
// CHECK:STDOUT:     %p.param: %ptr.c62 = value_param call_param0
// CHECK:STDOUT:     %.loc42_35: type = splice_block %ptr.loc42_35 [concrete = constants.%ptr.c62] {
// CHECK:STDOUT:       %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:       %ptr.loc42_35: type = ptr_type %Incomplete [concrete = constants.%ptr.c62]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.c62 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.590 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.590 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %.loc12_8: %A1.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %A1.elem = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %A1.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %.loc16_8: %A2.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc16_3: %A2.elem = var_pattern %.loc16_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %A2.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .a = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, file.%A2.decl [concrete = constants.%A2]
// CHECK:STDOUT:   %.loc20: %B2.elem.a92 = base_decl %A2.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc21_8: %B2.elem.4b2 = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc21_3: %B2.elem.4b2 = var_pattern %.loc21_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B2.elem.4b2 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b.618 [concrete = constants.%complete_type.92f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .A2 = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .b = %.loc21_8
// CHECK:STDOUT:   extend %A2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertUnrelated(%p.param_patt: %ptr.afe) -> %ptr.678 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.afe = name_ref p, %p
// CHECK:STDOUT:   %.loc31_46: %ptr.678 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertIncomplete(%p.param_patt: %ptr.c62) -> %ptr.590 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.c62 = name_ref p, %p
// CHECK:STDOUT:   %.loc42_55: %ptr.590 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
