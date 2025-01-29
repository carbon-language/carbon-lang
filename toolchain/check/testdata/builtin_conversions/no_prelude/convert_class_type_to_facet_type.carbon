// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_facet_type.carbon

interface Goat {}

class Ginger {}
impl Ginger as Goat {}

fn WalkGoat(g:! Goat) {}

fn F() {
  WalkGoat(Ginger);
}

// CHECK:STDOUT: --- convert_class_type_to_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Goat.type: type = facet_type <@Goat> [template]
// CHECK:STDOUT:   %Self: %Goat.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ginger: type = class_type @Ginger [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %g: %Goat.type = bind_symbolic_name g, 0 [symbolic]
// CHECK:STDOUT:   %g.patt: %Goat.type = symbolic_binding_pattern g, 0 [symbolic]
// CHECK:STDOUT:   %WalkGoat.type: type = fn_type @WalkGoat [template]
// CHECK:STDOUT:   %WalkGoat: %WalkGoat.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Goat.facet: %Goat.type = facet_value %Ginger, %impl_witness [template]
// CHECK:STDOUT:   %WalkGoat.specific_fn: <specific function> = specific_function %WalkGoat, @WalkGoat(%Goat.facet) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .Ginger = %Ginger.decl
// CHECK:STDOUT:     .WalkGoat = %WalkGoat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = interface_decl @Goat [template = constants.%Goat.type] {} {}
// CHECK:STDOUT:   %Ginger.decl: type = class_decl @Ginger [template = constants.%Ginger] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Ginger.ref: type = name_ref Ginger, file.%Ginger.decl [template = constants.%Ginger]
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template = constants.%impl_witness]
// CHECK:STDOUT:   %WalkGoat.decl: %WalkGoat.type = fn_decl @WalkGoat [template = constants.%WalkGoat] {
// CHECK:STDOUT:     %g.patt.loc16_13.1: %Goat.type = symbolic_binding_pattern g, 0 [symbolic = %g.patt.loc16_13.2 (constants.%g.patt)]
// CHECK:STDOUT:     %g.param_patt: %Goat.type = value_param_pattern %g.patt.loc16_13.1, runtime_param<none> [symbolic = %g.patt.loc16_13.2 (constants.%g.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %g.param: %Goat.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat.type]
// CHECK:STDOUT:     %g.loc16_13.1: %Goat.type = bind_symbolic_name g, 0, %g.param [symbolic = %g.loc16_13.2 (constants.%g)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Goat {
// CHECK:STDOUT:   %Self: %Goat.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Ginger.ref as %Goat.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Ginger {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Ginger
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WalkGoat(%g.loc16_13.1: %Goat.type) {
// CHECK:STDOUT:   %g.loc16_13.2: %Goat.type = bind_symbolic_name g, 0 [symbolic = %g.loc16_13.2 (constants.%g)]
// CHECK:STDOUT:   %g.patt.loc16_13.2: %Goat.type = symbolic_binding_pattern g, 0 [symbolic = %g.patt.loc16_13.2 (constants.%g.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%g.param_patt: %Goat.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WalkGoat.ref: %WalkGoat.type = name_ref WalkGoat, file.%WalkGoat.decl [template = constants.%WalkGoat]
// CHECK:STDOUT:   %Ginger.ref: type = name_ref Ginger, file.%Ginger.decl [template = constants.%Ginger]
// CHECK:STDOUT:   %Goat.facet: %Goat.type = facet_value %Ginger.ref, constants.%impl_witness [template = constants.%Goat.facet]
// CHECK:STDOUT:   %.loc19: %Goat.type = converted %Ginger.ref, %Goat.facet [template = constants.%Goat.facet]
// CHECK:STDOUT:   %WalkGoat.specific_fn: <specific function> = specific_function %WalkGoat.ref, @WalkGoat(constants.%Goat.facet) [template = constants.%WalkGoat.specific_fn]
// CHECK:STDOUT:   %WalkGoat.call: init %empty_tuple.type = call %WalkGoat.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkGoat(constants.%g) {
// CHECK:STDOUT:   %g.loc16_13.2 => constants.%g
// CHECK:STDOUT:   %g.patt.loc16_13.2 => constants.%g
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkGoat(constants.%Goat.facet) {
// CHECK:STDOUT:   %g.loc16_13.2 => constants.%Goat.facet
// CHECK:STDOUT:   %g.patt.loc16_13.2 => constants.%Goat.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
