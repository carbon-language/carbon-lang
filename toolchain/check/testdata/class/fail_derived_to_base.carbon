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
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: note: type `B2*` does not implement interface `ImplicitAs(A1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+6]]:47: error: cannot implicitly convert from `Incomplete*` to `A2*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:47: note: type `Incomplete*` does not implement interface `ImplicitAs(A2*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %A1, %i32 [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %struct_type.a [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %A2, %i32 [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B2, %A2 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B2, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.b.1: type = struct_type {.base: %A2, .b: %i32} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %struct_type.base.b.1 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %B2 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %A1 [template]
// CHECK:STDOUT:   %ConvertUnrelated.type: type = fn_type @ConvertUnrelated [template]
// CHECK:STDOUT:   %ConvertUnrelated: %ConvertUnrelated.type = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.35: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %.36: type = ptr_type %A2 [template]
// CHECK:STDOUT:   %ConvertIncomplete.type: type = fn_type @ConvertIncomplete [template]
// CHECK:STDOUT:   %ConvertIncomplete: %ConvertIncomplete.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:     %p.patt: %.8 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.8 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.9 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B2.ref: type = name_ref B2, file.%B2.decl [template = constants.%B2]
// CHECK:STDOUT:     %.loc31_26: type = ptr_type %B2 [template = constants.%.8]
// CHECK:STDOUT:     %A1.ref: type = name_ref A1, file.%A1.decl [template = constants.%A1]
// CHECK:STDOUT:     %.loc31_34: type = ptr_type %A1 [template = constants.%.9]
// CHECK:STDOUT:     %p.param: %.8 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.8 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.9 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %ConvertIncomplete.decl: %ConvertIncomplete.type = fn_decl @ConvertIncomplete [template = constants.%ConvertIncomplete] {
// CHECK:STDOUT:     %p.patt: %.35 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.35 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.36 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.36 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc41_35: type = ptr_type %Incomplete [template = constants.%.35]
// CHECK:STDOUT:     %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:     %.loc41_43: type = ptr_type %A2 [template = constants.%.36]
// CHECK:STDOUT:     %p.param: %.35 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.35 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.36 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.1 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %struct_type.a [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   complete_type_witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_signed, %.loc16_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_8: %.3 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %struct_type.a [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .a = %.loc16_8
// CHECK:STDOUT:   complete_type_witness = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:   %.loc20: %.5 = base_decl %A2.ref, element0 [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_signed, %.loc21_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_8: %.6 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %struct_type.base.b.1 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .b = %.loc21_8
// CHECK:STDOUT:   extend %A2.ref
// CHECK:STDOUT:   complete_type_witness = %.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertUnrelated(%p.param_patt: %.8) -> %.9 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_46: %.9 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertIncomplete(%p.param_patt: %.35) -> %.36 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.35 = name_ref p, %p
// CHECK:STDOUT:   %.loc41_55: %.36 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
