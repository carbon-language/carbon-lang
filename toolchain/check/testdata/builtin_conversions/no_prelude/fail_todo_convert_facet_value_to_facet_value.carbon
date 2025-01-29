// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_todo_convert_facet_value_to_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_todo_convert_facet_value_to_facet_value.carbon

interface Eats {}
interface Animal {}

// TODO: This may be rejected in the future.
// https://github.com/carbon-language/carbon-lang/issues/4853
impl Animal as Eats {}

fn Feed(e:! Eats) {}

class Goat {}
impl Goat as Animal {}

fn F() {
  // CHECK:STDERR: fail_todo_convert_facet_value_to_facet_value.carbon:[[@LINE+14]]:3: error: semantics TODO: `Facet value converting to facet value` [SemanticsTodo]
  // CHECK:STDERR:   Feed(Goat as Animal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_convert_facet_value_to_facet_value.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Feed(e:! Eats) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_convert_facet_value_to_facet_value.carbon:[[@LINE+7]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   Feed(Goat as Animal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_convert_facet_value_to_facet_value.carbon:[[@LINE-16]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Feed(e:! Eats) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Feed(Goat as Animal);
}

// CHECK:STDOUT: --- fail_todo_convert_facet_value_to_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [template]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [template]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %e: %Eats.type = bind_symbolic_name e, 0 [symbolic]
// CHECK:STDOUT:   %e.patt: %Eats.type = symbolic_binding_pattern e, 0 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [template]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [template]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [template = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [template = constants.%Animal.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [template = constants.%Animal.type]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [template = constants.%Eats.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness () [template = constants.%impl_witness]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [template = constants.%Feed] {
// CHECK:STDOUT:     %e.patt.loc18_9.1: %Eats.type = symbolic_binding_pattern e, 0 [symbolic = %e.patt.loc18_9.2 (constants.%e.patt)]
// CHECK:STDOUT:     %e.param_patt: %Eats.type = value_param_pattern %e.patt.loc18_9.1, runtime_param<none> [symbolic = %e.patt.loc18_9.2 (constants.%e.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Eats.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [template = constants.%Eats.type]
// CHECK:STDOUT:     %e.loc18_9.1: %Eats.type = bind_symbolic_name e, 0, %e.param [symbolic = %e.loc18_9.2 (constants.%e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [template = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [template = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc21: <witness> = impl_witness () [template = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Animal.ref as %Eats.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc21
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
// CHECK:STDOUT: generic fn @Feed(%e.loc18_9.1: %Eats.type) {
// CHECK:STDOUT:   %e.loc18_9.2: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc18_9.2 (constants.%e)]
// CHECK:STDOUT:   %e.patt.loc18_9.2: %Eats.type = symbolic_binding_pattern e, 0 [symbolic = %e.patt.loc18_9.2 (constants.%e.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param_patt: %Eats.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [template = constants.%Feed]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [template = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref: type = name_ref Animal, file.%Animal.decl [template = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat.ref, constants.%impl_witness [template = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc38_13: %Animal.type = converted %Goat.ref, %Animal.facet [template = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc38_22: %Eats.type = converted %.loc38_13, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%e) {
// CHECK:STDOUT:   %e.loc18_9.2 => constants.%e
// CHECK:STDOUT:   %e.patt.loc18_9.2 => constants.%e
// CHECK:STDOUT: }
// CHECK:STDOUT:
