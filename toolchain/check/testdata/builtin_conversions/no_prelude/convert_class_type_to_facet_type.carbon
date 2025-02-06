// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_facet_type.carbon

interface Animal {}

class Goat {}
impl Goat as Animal {}

fn WalkAnimal(A:! Animal) {}

fn F() {
  WalkAnimal(Goat);
}

// CHECK:STDOUT: --- convert_class_type_to_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [template]
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %A: %Animal.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.patt: %Animal.type = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %WalkAnimal.type: type = fn_type @WalkAnimal [template]
// CHECK:STDOUT:   %WalkAnimal: %WalkAnimal.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, %impl_witness [template]
// CHECK:STDOUT:   %WalkAnimal.specific_fn: <specific function> = specific_function %WalkAnimal, @WalkAnimal(%Animal.facet) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .WalkAnimal = %WalkAnimal.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [template = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [template = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [template = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template = constants.%impl_witness]
// CHECK:STDOUT:   %WalkAnimal.decl: %WalkAnimal.type = fn_decl @WalkAnimal [template = constants.%WalkAnimal] {
// CHECK:STDOUT:     %A.patt.loc16_15.1: %Animal.type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc16_15.2 (constants.%A.patt)]
// CHECK:STDOUT:     %A.param_patt: %Animal.type = value_param_pattern %A.patt.loc16_15.1, runtime_param<none> [symbolic = %A.patt.loc16_15.2 (constants.%A.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.param: %Animal.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [template = constants.%Animal.type]
// CHECK:STDOUT:     %A.loc16_15.1: %Animal.type = bind_symbolic_name A, 0, %A.param [symbolic = %A.loc16_15.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WalkAnimal(%A.loc16_15.1: %Animal.type) {
// CHECK:STDOUT:   %A.loc16_15.2: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc16_15.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc16_15.2: %Animal.type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc16_15.2 (constants.%A.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%A.param_patt: %Animal.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WalkAnimal.ref: %WalkAnimal.type = name_ref WalkAnimal, file.%WalkAnimal.decl [template = constants.%WalkAnimal]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat.ref, constants.%impl_witness [template = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc19: %Animal.type = converted %Goat.ref, %Animal.facet [template = constants.%Animal.facet]
// CHECK:STDOUT:   %WalkAnimal.specific_fn: <specific function> = specific_function %WalkAnimal.ref, @WalkAnimal(constants.%Animal.facet) [template = constants.%WalkAnimal.specific_fn]
// CHECK:STDOUT:   %WalkAnimal.call: init %empty_tuple.type = call %WalkAnimal.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkAnimal(constants.%A) {
// CHECK:STDOUT:   %A.loc16_15.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc16_15.2 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %A.loc16_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %A.patt.loc16_15.2 => constants.%Animal.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
