// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon

// --- core.carbon

package Core;

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

// --- fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon

library "[[@TEST_NAME]]";

import Core;

class Grass {}

interface Animal {}
interface Eats(Food:! type) {}

fn Feed[Food:! type, T:! Eats(Food)](e: T, food: Food) {}
// CHECK:STDERR: fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon:[[@LINE+10]]:62: error: cannot implicitly convert from `type` to `Eats(Food)` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn HandleAnimal[T:! Animal, Food:! type](a: T, food: Food) { Feed(a, food); }
// CHECK:STDERR:                                                              ^~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon:[[@LINE+7]]:62: note: type `type` does not implement interface `Core.ImplicitAs(Eats(Food))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn HandleAnimal[T:! Animal, Food:! type](a: T, food: Food) { Feed(a, food); }
// CHECK:STDERR:                                                              ^~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn Feed[Food:! type, T:! Eats(Food)](e: T, food: Food) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleAnimal[T:! Animal, Food:! type](a: T, food: Food) { Feed(a, food); }

class Goat {}
impl Goat as Animal {}

impl forall [T:! Animal] T as Eats(Grass) {}

fn F() {
  HandleAnimal({} as Goat, {} as Grass);
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc4_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_22.1, runtime_param<none> [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc4_22.1: type) {
// CHECK:STDOUT:   %T.loc4_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc4_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T.loc4_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_32.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc4_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %.loc5_20.1: type = splice_block %.loc5_20.3 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc5_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_20.3: type = converted %Self.ref, %Self.as_type.loc5_20.2 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_32.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc5_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%T.loc4_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type)]() -> @Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc4_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_convert_facet_value_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Food.8b3: type = bind_symbolic_name Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.patt.e01: type = symbolic_binding_pattern Food, 0 [symbolic]
// CHECK:STDOUT:   %Eats.type.ba2: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.6c0: type = facet_type <@Eats, @Eats(%Food.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.4eb: %Eats.type.6c0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.4eb: %Eats.type.6c0 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.41f: %Eats.type.6c0 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.7b9: type = facet_access_type %T.4eb [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c74: <witness> = require_complete_type %T.as_type.7b9 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %Food.8b3 [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.a9c: %Animal.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Food.336: type = bind_symbolic_name Food, 1 [symbolic]
// CHECK:STDOUT:   %Food.patt.7a9: type = symbolic_binding_pattern Food, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %Food.336 [symbolic]
// CHECK:STDOUT:   %Eats.type.3ec: type = facet_type <@Eats, @Eats(%Food.336)> [symbolic]
// CHECK:STDOUT:   %require_complete.588: <witness> = require_complete_type %Eats.type.3ec [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.dd1: type = facet_type <@ImplicitAs, @ImplicitAs(%Eats.type.3ec)> [symbolic]
// CHECK:STDOUT:   %require_complete.959: <witness> = require_complete_type %ImplicitAs.type.dd1 [symbolic]
// CHECK:STDOUT:   %Convert.type.d69: type = fn_type @Convert, @ImplicitAs(%Eats.type.3ec) [symbolic]
// CHECK:STDOUT:   %Convert.561: %Convert.type.d69 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.163: type = assoc_entity_type %ImplicitAs.type.dd1 [symbolic]
// CHECK:STDOUT:   %assoc0.676: %ImplicitAs.assoc_type.163 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %Eats.type.1ae: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %impl_witness.8fd: <witness> = impl_witness (), @impl.2(%T.fd4) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, %impl_witness.1bc [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Grass) [concrete]
// CHECK:STDOUT:   %complete_type.004: <witness> = complete_type_witness %Eats.type.1ae [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.849: type = facet_type <@ImplicitAs, @ImplicitAs(%Eats.type.1ae)> [concrete]
// CHECK:STDOUT:   %complete_type.eef: <witness> = complete_type_witness %ImplicitAs.type.849 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.f6b058.1: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst26 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc5_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.f6b058.2: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst26 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc5_32, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Grass = %Grass.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Grass.decl: type = class_decl @Grass [concrete = constants.%Grass] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Eats.decl: %Eats.type.ba2 = interface_decl @Eats [concrete = constants.%Eats.generic] {
// CHECK:STDOUT:     %Food.patt.loc9_16.1: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc9_16.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:     %Food.param_patt: type = value_param_pattern %Food.patt.loc9_16.1, runtime_param<none> [symbolic = %Food.patt.loc9_16.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Food.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Food.loc9_16.1: type = bind_symbolic_name Food, 0, %Food.param [symbolic = %Food.loc9_16.2 (constants.%Food.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt.loc11_9.1: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc11_9.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:     %Food.param_patt: type = value_param_pattern %Food.patt.loc11_9.1, runtime_param<none> [symbolic = %Food.patt.loc11_9.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc11_22.1: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt.41f)]
// CHECK:STDOUT:     %T.param_patt: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0) = value_param_pattern %T.patt.loc11_22.1, runtime_param<none> [symbolic = %T.patt.loc11_22.2 (constants.%T.patt.41f)]
// CHECK:STDOUT:     %e.patt: @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9) = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9) = value_param_pattern %e.patt, runtime_param0
// CHECK:STDOUT:     %food.patt: @Feed.%Food.loc11_9.2 (%Food.8b3) = binding_pattern food
// CHECK:STDOUT:     %food.param_patt: @Feed.%Food.loc11_9.2 (%Food.8b3) = value_param_pattern %food.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Food.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Food.loc11_9.1: type = bind_symbolic_name Food, 0, %Food.param [symbolic = %Food.loc11_9.2 (constants.%Food.8b3)]
// CHECK:STDOUT:     %T.param: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc11_35: type = splice_block %Eats.type.loc11_35.1 [symbolic = %Eats.type.loc11_35.2 (constants.%Eats.type.6c0)] {
// CHECK:STDOUT:       %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc11_31: type = name_ref Food, %Food.loc11_9.1 [symbolic = %Food.loc11_9.2 (constants.%Food.8b3)]
// CHECK:STDOUT:       %Eats.type.loc11_35.1: type = facet_type <@Eats, @Eats(constants.%Food.8b3)> [symbolic = %Eats.type.loc11_35.2 (constants.%Eats.type.6c0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_22.1: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0) = bind_symbolic_name T, 1, %T.param [symbolic = %T.loc11_22.2 (constants.%T.4eb)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9) = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_41.1: type = splice_block %.loc11_41.2 [symbolic = %T.as_type.loc11_41.2 (constants.%T.as_type.7b9)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0) = name_ref T, %T.loc11_22.1 [symbolic = %T.loc11_22.2 (constants.%T.4eb)]
// CHECK:STDOUT:       %T.as_type.loc11_41.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_41.2 (constants.%T.as_type.7b9)]
// CHECK:STDOUT:       %.loc11_41.2: type = converted %T.ref, %T.as_type.loc11_41.1 [symbolic = %T.as_type.loc11_41.2 (constants.%T.as_type.7b9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9) = bind_name e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.loc11_9.2 (%Food.8b3) = value_param runtime_param1
// CHECK:STDOUT:     %Food.ref.loc11_50: type = name_ref Food, %Food.loc11_9.1 [symbolic = %Food.loc11_9.2 (constants.%Food.8b3)]
// CHECK:STDOUT:     %food: @Feed.%Food.loc11_9.2 (%Food.8b3) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt.loc22_17.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_17.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:     %T.param_patt: %Animal.type = value_param_pattern %T.patt.loc22_17.1, runtime_param<none> [symbolic = %T.patt.loc22_17.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:     %Food.patt.loc22_29.1: type = symbolic_binding_pattern Food, 1 [symbolic = %Food.patt.loc22_29.2 (constants.%Food.patt.7a9)]
// CHECK:STDOUT:     %Food.param_patt: type = value_param_pattern %Food.patt.loc22_29.1, runtime_param<none> [symbolic = %Food.patt.loc22_29.2 (constants.%Food.patt.7a9)]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%Food.loc22_29.2 (%Food.336) = binding_pattern food
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%Food.loc22_29.2 (%Food.336) = value_param_pattern %food.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Animal.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc22_17.1: %Animal.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %Food.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Food.loc22_29.1: type = bind_symbolic_name Food, 1, %Food.param [symbolic = %Food.loc22_29.2 (constants.%Food.336)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) = value_param runtime_param0
// CHECK:STDOUT:     %.loc22_45.1: type = splice_block %.loc22_45.2 [symbolic = %T.as_type.loc22_45.2 (constants.%T.as_type.2ad)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc22_17.1 [symbolic = %T.loc22_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc22_45.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_45.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:       %.loc22_45.2: type = converted %T.ref, %T.as_type.loc22_45.1 [symbolic = %T.as_type.loc22_45.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) = bind_name a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.loc22_29.2 (%Food.336) = value_param runtime_param1
// CHECK:STDOUT:     %Food.ref: type = name_ref Food, %Food.loc22_29.1 [symbolic = %Food.loc22_29.2 (constants.%Food.336)]
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.loc22_29.2 (%Food.336) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc27_14.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:     %T.param_patt: %Animal.type = value_param_pattern %T.patt.loc27_14.1, runtime_param<none> [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc27_14.1 [symbolic = %T.loc27_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %T.as_type.loc27_26.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc27_26.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc27: type = converted %T.ref, %T.as_type.loc27_26.1 [symbolic = %T.as_type.loc27_26.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:     %Eats.type: type = facet_type <@Eats, @Eats(constants.%Grass)> [concrete = constants.%Eats.type.1ae]
// CHECK:STDOUT:     %T.param: %Animal.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc27_14.1: %Animal.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc27_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (), @impl.2(constants.%T.fd4) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.8fd)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
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
// CHECK:STDOUT: generic interface @Eats(%Food.loc9_16.1: type) {
// CHECK:STDOUT:   %Food.loc9_16.2: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc9_16.2 (constants.%Food.8b3)]
// CHECK:STDOUT:   %Food.patt.loc9_16.2: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc9_16.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc9_16.2)> [symbolic = %Eats.type (constants.%Eats.type.6c0)]
// CHECK:STDOUT:   %Self.2: %Eats.type.6c0 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Eats.%Eats.type (%Eats.type.6c0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.f6b058.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc27_14.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc27_14.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.patt.loc27_14.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:   %T.as_type.loc27_26.2: type = facet_access_type %T.loc27_14.2 [symbolic = %T.as_type.loc27_26.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc27_14.2) [symbolic = %impl_witness (constants.%impl_witness.8fd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc27 as %Eats.type {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Grass {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Grass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%Food.loc11_9.1: type, %T.loc11_22.1: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0)) {
// CHECK:STDOUT:   %Food.loc11_9.2: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc11_9.2 (constants.%Food.8b3)]
// CHECK:STDOUT:   %Food.patt.loc11_9.2: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc11_9.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:   %Eats.type.loc11_35.2: type = facet_type <@Eats, @Eats(%Food.loc11_9.2)> [symbolic = %Eats.type.loc11_35.2 (constants.%Eats.type.6c0)]
// CHECK:STDOUT:   %T.loc11_22.2: %Eats.type.6c0 = bind_symbolic_name T, 1 [symbolic = %T.loc11_22.2 (constants.%T.4eb)]
// CHECK:STDOUT:   %T.patt.loc11_22.2: %Eats.type.6c0 = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt.41f)]
// CHECK:STDOUT:   %T.as_type.loc11_41.2: type = facet_access_type %T.loc11_22.2 [symbolic = %T.as_type.loc11_41.2 (constants.%T.as_type.7b9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_39: <witness> = require_complete_type @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9) [symbolic = %require_complete.loc11_39 (constants.%require_complete.c74)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type @Feed.%Food.loc11_9.2 (%Food.8b3) [symbolic = %require_complete.loc11_48 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%Food.param_patt: type, %T.param_patt: @Feed.%Eats.type.loc11_35.2 (%Eats.type.6c0)](%e.param_patt: @Feed.%T.as_type.loc11_41.2 (%T.as_type.7b9), %food.param_patt: @Feed.%Food.loc11_9.2 (%Food.8b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc22_17.1: %Animal.type, %Food.loc22_29.1: type) {
// CHECK:STDOUT:   %T.loc22_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.patt.loc22_17.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_17.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:   %Food.loc22_29.2: type = bind_symbolic_name Food, 1 [symbolic = %Food.loc22_29.2 (constants.%Food.336)]
// CHECK:STDOUT:   %Food.patt.loc22_29.2: type = symbolic_binding_pattern Food, 1 [symbolic = %Food.patt.loc22_29.2 (constants.%Food.patt.7a9)]
// CHECK:STDOUT:   %T.as_type.loc22_45.2: type = facet_access_type %T.loc22_17.2 [symbolic = %T.as_type.loc22_45.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22_43: <witness> = require_complete_type @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) [symbolic = %require_complete.loc22_43 (constants.%require_complete.234)]
// CHECK:STDOUT:   %require_complete.loc22_52: <witness> = require_complete_type @HandleAnimal.%Food.loc22_29.2 (%Food.336) [symbolic = %require_complete.loc22_52 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc22_29.2)> [symbolic = %Eats.type (constants.%Eats.type.3ec)]
// CHECK:STDOUT:   %require_complete.loc22_74.1: <witness> = require_complete_type @HandleAnimal.%Eats.type (%Eats.type.3ec) [symbolic = %require_complete.loc22_74.1 (constants.%require_complete.588)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Eats.type)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.dd1)]
// CHECK:STDOUT:   %require_complete.loc22_74.2: <witness> = require_complete_type @HandleAnimal.%ImplicitAs.type (%ImplicitAs.type.dd1) [symbolic = %require_complete.loc22_74.2 (constants.%require_complete.959)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: %Animal.type, %Food.param_patt: type](%a.param_patt: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad), %food.param_patt: @HandleAnimal.%Food.loc22_29.2 (%Food.336)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc22_45.2 (%T.as_type.2ad) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.loc22_29.2 (%Food.336) = name_ref food, %food
// CHECK:STDOUT:     %.loc22_74: @HandleAnimal.%Eats.type (%Eats.type.3ec) = converted constants.%T.as_type.2ad, <error> [concrete = <error>]
// CHECK:STDOUT:     %Feed.specific_fn: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.336, <error>) [concrete = <error>]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn(<error>, %food.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.f6b058.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%T (%T.8b3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc30_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc30_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc30_17.3: init %Goat = class_init (), %.loc30_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc30_17.4: ref %Goat = temporary %.loc30_17.2, %.loc30_17.3
// CHECK:STDOUT:   %.loc30_19.1: ref %Goat = converted %.loc30_17.1, %.loc30_17.4
// CHECK:STDOUT:   %.loc30_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:   %.loc30_29.2: ref %Grass = temporary_storage
// CHECK:STDOUT:   %.loc30_29.3: init %Grass = class_init (), %.loc30_29.2 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %.loc30_29.4: ref %Grass = temporary %.loc30_29.2, %.loc30_29.3
// CHECK:STDOUT:   %.loc30_31.1: ref %Grass = converted %.loc30_29.1, %.loc30_29.4
// CHECK:STDOUT:   %Animal.facet.loc30_39.1: %Animal.type = facet_value constants.%Goat, constants.%impl_witness.1bc [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc30_39.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc30_39.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc30_39.2: %Animal.type = facet_value constants.%Goat, constants.%impl_witness.1bc [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc30_39.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc30_39.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet, constants.%Grass) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc30_19.2: %Goat = bind_value %.loc30_19.1
// CHECK:STDOUT:   %.loc30_31.2: %Grass = bind_value %.loc30_31.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc30_19.2, %.loc30_31.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.8b3) {
// CHECK:STDOUT:   %Food.loc9_16.2 => constants.%Food.8b3
// CHECK:STDOUT:   %Food.patt.loc9_16.2 => constants.%Food.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(%Food.loc9_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.8b3, constants.%T.4eb) {
// CHECK:STDOUT:   %Food.loc11_9.2 => constants.%Food.8b3
// CHECK:STDOUT:   %Food.patt.loc11_9.2 => constants.%Food.8b3
// CHECK:STDOUT:   %Eats.type.loc11_35.2 => constants.%Eats.type.6c0
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%T.4eb
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T.4eb
// CHECK:STDOUT:   %T.as_type.loc11_41.2 => constants.%T.as_type.7b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(@Feed.%Food.loc11_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.fd4, constants.%Food.336) {
// CHECK:STDOUT:   %T.loc22_17.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.patt.loc22_17.2 => constants.%T.fd4
// CHECK:STDOUT:   %Food.loc22_29.2 => constants.%Food.336
// CHECK:STDOUT:   %Food.patt.loc22_29.2 => constants.%Food.336
// CHECK:STDOUT:   %T.as_type.loc22_45.2 => constants.%T.as_type.2ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.336) {
// CHECK:STDOUT:   %Food.loc9_16.2 => constants.%Food.336
// CHECK:STDOUT:   %Food.patt.loc9_16.2 => constants.%Food.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T.8b3, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Eats.type.3ec) {
// CHECK:STDOUT:   %T => constants.%Eats.type.3ec
// CHECK:STDOUT:   %T.patt => constants.%Eats.type.3ec
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.dd1
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.d69
// CHECK:STDOUT:   %Convert => constants.%Convert.561
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.163
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.676
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.336, <error>) {
// CHECK:STDOUT:   %Food.loc11_9.2 => constants.%Food.336
// CHECK:STDOUT:   %Food.patt.loc11_9.2 => constants.%Food.336
// CHECK:STDOUT:   %Eats.type.loc11_35.2 => constants.%Eats.type.3ec
// CHECK:STDOUT:   %T.loc11_22.2 => <error>
// CHECK:STDOUT:   %T.patt.loc11_22.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_41.2 => <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_39 => <error>
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(@HandleAnimal.%Food.loc22_29.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@HandleAnimal.%Eats.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc9_16.2 => constants.%Grass
// CHECK:STDOUT:   %Food.patt.loc9_16.2 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %Self.2 => constants.%Self.4eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T.fd4) {
// CHECK:STDOUT:   %T.loc27_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.patt.loc27_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.as_type.loc27_26.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc27_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Grass) {
// CHECK:STDOUT:   %T.loc22_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.patt.loc22_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc22_29.2 => constants.%Grass
// CHECK:STDOUT:   %Food.patt.loc22_29.2 => constants.%Grass
// CHECK:STDOUT:   %T.as_type.loc22_45.2 => constants.%Goat
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22_43 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc22_52 => constants.%complete_type.357
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %require_complete.loc22_74.1 => constants.%complete_type.004
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.849
// CHECK:STDOUT:   %require_complete.loc22_74.2 => constants.%complete_type.eef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Eats.type.1ae) {
// CHECK:STDOUT:   %T => constants.%Eats.type.1ae
// CHECK:STDOUT:   %T.patt => constants.%Eats.type.1ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
