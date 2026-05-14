// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/self_in_class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/self_in_class.carbon

interface DefaultConstructible {
  fn Make() -> Self;
}

class C {}

class A {
  impl C as DefaultConstructible {
    // `Self` here refers to `C`, not `A`.
    // TODO: Revisit this once #3714 is resolved.
    fn Make() -> Self { return {}; }
  }
}

// CHECK:STDOUT: --- self_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefaultConstructible.type: type = facet_type <@DefaultConstructible> [concrete]
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %.136: Core.Form = init_form %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.493: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.type.17e: type = fn_type @DefaultConstructible.WithSelf.Make, @DefaultConstructible.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.cfb: %DefaultConstructible.WithSelf.Make.type.17e = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultConstructible.assoc_type: type = assoc_entity_type @DefaultConstructible [concrete]
// CHECK:STDOUT:   %assoc0: %DefaultConstructible.assoc_type = assoc_entity element0, @DefaultConstructible.WithSelf.%DefaultConstructible.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %DefaultConstructible.impl_witness: <witness> = impl_witness @C.as.DefaultConstructible.impl.%DefaultConstructible.impl_witness_table [concrete]
// CHECK:STDOUT:   %.a69: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.DefaultConstructible.impl.Make.type: type = fn_type @C.as.DefaultConstructible.impl.Make [concrete]
// CHECK:STDOUT:   %C.as.DefaultConstructible.impl.Make: %C.as.DefaultConstructible.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultConstructible.facet: %DefaultConstructible.type = facet_value %C, (%DefaultConstructible.impl_witness) [concrete]
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.type.0cb: type = fn_type @DefaultConstructible.WithSelf.Make, @DefaultConstructible.WithSelf(%DefaultConstructible.facet) [concrete]
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.49d: %DefaultConstructible.WithSelf.Make.type.0cb = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .DefaultConstructible = %DefaultConstructible.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructible.decl: type = interface_decl @DefaultConstructible [concrete = constants.%DefaultConstructible.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultConstructible {
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.decl = interface_with_self_decl @DefaultConstructible [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.decl: @DefaultConstructible.WithSelf.%DefaultConstructible.WithSelf.Make.type (%DefaultConstructible.WithSelf.Make.type.17e) = fn_decl @DefaultConstructible.WithSelf.Make [symbolic = @DefaultConstructible.WithSelf.%DefaultConstructible.WithSelf.Make (constants.%DefaultConstructible.WithSelf.Make.cfb)] {
// CHECK:STDOUT:     %return.param_patt: @DefaultConstructible.WithSelf.Make.%pattern_type (%pattern_type.493) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @DefaultConstructible.WithSelf.Make.%pattern_type (%pattern_type.493) = return_slot_pattern %return.param_patt, %.loc16_16.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %DefaultConstructible.type = name_ref Self, @DefaultConstructible.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc16_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc16_16.2: type = converted %Self.ref, %Self.as_type.loc16_16.2 [symbolic = %Self.as_type.loc16_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc16_16.3: Core.Form = init_form %.loc16_16.2 [symbolic = %.loc16_16.1 (constants.%.136)]
// CHECK:STDOUT:     %return.param: ref @DefaultConstructible.WithSelf.Make.%Self.as_type.loc16_16.1 (%Self.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @DefaultConstructible.WithSelf.Make.%Self.as_type.loc16_16.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %DefaultConstructible.assoc_type = assoc_entity element0, %DefaultConstructible.WithSelf.Make.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Make = @DefaultConstructible.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@DefaultConstructible.WithSelf.%DefaultConstructible.WithSelf.Make.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.DefaultConstructible.impl: %C.ref as %DefaultConstructible.ref {
// CHECK:STDOUT:   %C.as.DefaultConstructible.impl.Make.decl: %C.as.DefaultConstructible.impl.Make.type = fn_decl @C.as.DefaultConstructible.impl.Make [concrete = constants.%C.as.DefaultConstructible.impl.Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern %return.param_patt, %Self.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.DefaultConstructible.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %.loc25_18: Core.Form = init_form %Self.ref [concrete = constants.%.a69]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructible.impl_witness_table = impl_witness_table (%C.as.DefaultConstructible.impl.Make.decl), @C.as.DefaultConstructible.impl [concrete]
// CHECK:STDOUT:   %DefaultConstructible.impl_witness: <witness> = impl_witness %DefaultConstructible.impl_witness_table [concrete = constants.%DefaultConstructible.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %C.as.DefaultConstructible.impl.Make.decl
// CHECK:STDOUT:   extend %DefaultConstructible.ref
// CHECK:STDOUT:   witness = %DefaultConstructible.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @C.as.DefaultConstructible.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %DefaultConstructible.ref: type = name_ref DefaultConstructible, file.%DefaultConstructible.decl [concrete = constants.%DefaultConstructible.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .DefaultConstructible = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DefaultConstructible.WithSelf.Make(@DefaultConstructible.%Self: %DefaultConstructible.type) {
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc16_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %.loc16_16.1: Core.Form = init_form %Self.as_type.loc16_16.1 [symbolic = %.loc16_16.1 (constants.%.136)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_16.1 [symbolic = %pattern_type (constants.%pattern_type.493)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @DefaultConstructible.WithSelf.Make.%Self.as_type.loc16_16.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.DefaultConstructible.impl.Make() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc25_33.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc25_34: init %C = converted %.loc25_33.1, %.loc25_33.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc25_34 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultConstructible.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.type => constants.%DefaultConstructible.WithSelf.Make.type.17e
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make => constants.%DefaultConstructible.WithSelf.Make.cfb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultConstructible.WithSelf.Make(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc16_16.1 => constants.%Self.as_type
// CHECK:STDOUT:   %.loc16_16.1 => constants.%.136
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.493
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultConstructible.WithSelf(constants.%DefaultConstructible.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%DefaultConstructible.facet
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make.type => constants.%DefaultConstructible.WithSelf.Make.type.0cb
// CHECK:STDOUT:   %DefaultConstructible.WithSelf.Make => constants.%DefaultConstructible.WithSelf.Make.49d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultConstructible.WithSelf.Make(constants.%DefaultConstructible.facet) {
// CHECK:STDOUT:   %Self => constants.%DefaultConstructible.facet
// CHECK:STDOUT:   %Self.as_type.loc16_16.1 => constants.%C
// CHECK:STDOUT:   %.loc16_16.1 => constants.%.a69
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
