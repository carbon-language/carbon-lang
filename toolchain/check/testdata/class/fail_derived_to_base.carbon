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

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:38: error: cannot implicitly convert from `B2*` to `A1*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: note: type `B2*` does not implement interface `Core.ImplicitAs(A1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+6]]:47: error: cannot implicitly convert from `Incomplete*` to `A2*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:47: note: type `Incomplete*` does not implement interface `Core.ImplicitAs(A2*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %A1.elem: type = unbound_element_type %A1, %i32 [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.2976: <witness> = complete_type_witness %struct_type.a [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %i32 [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %B2.elem.a92e: type = unbound_element_type %B2, %A2 [template]
// CHECK:STDOUT:   %B2.elem.9b97: type = unbound_element_type %B2, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.b.a128: type = struct_type {.base: %A2, .b: %i32} [template]
// CHECK:STDOUT:   %complete_type.3048: <witness> = complete_type_witness %struct_type.base.b.a128 [template]
// CHECK:STDOUT:   %ptr.afe4: type = ptr_type %B2 [template]
// CHECK:STDOUT:   %ptr.678d: type = ptr_type %A1 [template]
// CHECK:STDOUT:   %ConvertUnrelated.type: type = fn_type @ConvertUnrelated [template]
// CHECK:STDOUT:   %ConvertUnrelated: %ConvertUnrelated.type = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %ptr.c620: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %ptr.5907: type = ptr_type %A2 [template]
// CHECK:STDOUT:   %ConvertIncomplete.type: type = fn_type @ConvertIncomplete [template]
// CHECK:STDOUT:   %ConvertIncomplete: %ConvertIncomplete.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .ConvertUnrelated = %ConvertUnrelated.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .ConvertIncomplete = %ConvertIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [template = constants.%A1] {} {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [template = constants.%A2] {} {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [template = constants.%B2] {} {}
// CHECK:STDOUT:   %ConvertUnrelated.decl: %ConvertUnrelated.type = fn_decl @ConvertUnrelated [template = constants.%ConvertUnrelated] {
// CHECK:STDOUT:     %p.patt: %ptr.afe4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.afe4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.678d = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.678d = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A1.ref: type = name_ref A1, file.%A1.decl [template = constants.%A1]
// CHECK:STDOUT:     %ptr.loc31_34: type = ptr_type %A1 [template = constants.%ptr.678d]
// CHECK:STDOUT:     %p.param: %ptr.afe4 = value_param runtime_param0
// CHECK:STDOUT:     %.loc31_26: type = splice_block %ptr.loc31_26 [template = constants.%ptr.afe4] {
// CHECK:STDOUT:       %B2.ref: type = name_ref B2, file.%B2.decl [template = constants.%B2]
// CHECK:STDOUT:       %ptr.loc31_26: type = ptr_type %B2 [template = constants.%ptr.afe4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.afe4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.678d = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.678d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %ConvertIncomplete.decl: %ConvertIncomplete.type = fn_decl @ConvertIncomplete [template = constants.%ConvertIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.c620 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.c620 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.5907 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.5907 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:     %ptr.loc41_43: type = ptr_type %A2 [template = constants.%ptr.5907]
// CHECK:STDOUT:     %p.param: %ptr.c620 = value_param runtime_param0
// CHECK:STDOUT:     %.loc41_35: type = splice_block %ptr.loc41_35 [template = constants.%ptr.c620] {
// CHECK:STDOUT:       %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:       %ptr.loc41_35: type = ptr_type %Incomplete [template = constants.%ptr.c620]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.c620 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.5907 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.5907 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %.loc12: %A1.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [template = constants.%complete_type.2976]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %.loc16: %A2.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [template = constants.%complete_type.2976]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:   %.loc20: %B2.elem.a92e = base_decl %A2.ref, element0 [template]
// CHECK:STDOUT:   %.loc21: %B2.elem.9b97 = field_decl b, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b.a128 [template = constants.%complete_type.3048]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .b = %.loc21
// CHECK:STDOUT:   extend %A2.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertUnrelated(%p.param_patt: %ptr.afe4) -> %ptr.678d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.afe4 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_46: %ptr.678d = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertIncomplete(%p.param_patt: %ptr.c620) -> %ptr.5907 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.c620 = name_ref p, %p
// CHECK:STDOUT:   %.loc41_55: %ptr.5907 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
