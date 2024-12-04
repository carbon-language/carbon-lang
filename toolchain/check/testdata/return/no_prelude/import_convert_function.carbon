// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/no_prelude/import_convert_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/no_prelude/import_convert_function.carbon

// --- core.carbon

// A minimized prelude intended to contain as few functions as possible.
package Core;

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

// --- library.carbon

package P library "[[@TEST_NAME]]";

import Core;

class C(N:! i32) {}
class D { var n: i32; var m: i32; }
fn Make() -> D { return {.n = 0, .m = 0}; }

impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";
import Core;

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(n: i32) -> P.D {
  if (false) { return {} as P.C(0); }
  if (false) { return {} as P.C(1); }
  if (false) { return {} as P.C(2); }
  if (false) { return {} as P.C(3); }
  if (false) { return {} as P.C(4); }
  if (false) { return {} as P.C(5); }
  if (false) { return {} as P.C(6); }
  if (false) { return {} as P.C(7); }
  return P.Make();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2 = facet_value Core.IntLiteral, Core.IntLiteral [symbolic]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int_literal.make_type, %.loc6_22.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc8_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_22.1, runtime_param<invalid> [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %int_literal.make_type, %.loc12_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %int.make_type_signed, %.loc12_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc8_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc8_22.2) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc9_32.2: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_20.1: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc9_20.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_20.2: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_32.1: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc9_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc12_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed, %.loc13_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc8_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc8_22.2) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0.loc9_32.2 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [template]
// CHECK:STDOUT:   %struct_type.n.m.1: type = struct_type {.n: %i32, .m: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.n.m.1 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct_type.n.m.2: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.2, %int_0.2) [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%int_0.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.3: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %assoc0.4: %Convert.assoc_type.3 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %ImplicitAs.facet.1: %ImplicitAs.type.2 = facet_value %C.2, %C.2 [symbolic]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%int_1.2) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.2: %ImplicitAs.type.2 = facet_value %C.3, %C.3 [symbolic]
// CHECK:STDOUT:   %interface.3: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%int_2.2) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.3: %ImplicitAs.type.2 = facet_value %C.4, %C.4 [symbolic]
// CHECK:STDOUT:   %interface.4: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%int_3.2) [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.4: %ImplicitAs.type.2 = facet_value %C.5, %C.5 [symbolic]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_4.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%int_4.2) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.5: %ImplicitAs.type.2 = facet_value %C.6, %C.6 [symbolic]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_5.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%int_5.2) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.6: %ImplicitAs.type.2 = facet_value %C.7, %C.7 [symbolic]
// CHECK:STDOUT:   %interface.7: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_6.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%int_6.2) [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.7: %ImplicitAs.type.2 = facet_value %C.8, %C.8 [symbolic]
// CHECK:STDOUT:   %interface.8: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %int_7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_7.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%int_7.2) [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.8: %ImplicitAs.type.2 = facet_value %C.9, %C.9 [symbolic]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.12) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//default, inst+30, loaded [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = import_ref Core//default, inst+60, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%Convert.type (%Convert.type.1) = import_ref Core//default, inst+52, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//default, inst+71, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//default, inst+79, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//default, inst+103, loaded [template = constants.%interface.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_13.2: type = converted %int.make_type_signed, %.loc6_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_0, %.loc10_9.1 [template = constants.%int_0.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.2) [template = constants.%C.2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_1, %.loc11_9.1 [template = constants.%int_1.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.2) [template = constants.%C.3]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_2, %.loc12_9.1 [template = constants.%int_2.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.2) [template = constants.%C.4]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_3, %impl.elem0 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_3) [template = constants.%int_3.2]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_3.2]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_3, %.loc13_9.1 [template = constants.%int_3.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.2) [template = constants.%C.5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_4, %impl.elem0 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_4) [template = constants.%int_4.2]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_4.2]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_4, %.loc14_9.1 [template = constants.%int_4.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.2) [template = constants.%C.6]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_5, %.loc15_9.1 [template = constants.%int_5.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.2) [template = constants.%C.7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_6, %impl.elem0 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_6) [template = constants.%int_6.2]
// CHECK:STDOUT:     %.loc16_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_6.2]
// CHECK:STDOUT:     %.loc16_9.2: %i32 = converted %int_6, %.loc16_9.1 [template = constants.%int_6.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.2) [template = constants.%C.8]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_7, %impl.elem0 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.bound(%int_7) [template = constants.%int_7.2]
// CHECK:STDOUT:     %.loc17_9.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_7.2]
// CHECK:STDOUT:     %.loc17_9.2: %i32 = converted %int_7, %.loc17_9.1 [template = constants.%int_7.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.2) [template = constants.%C.9]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, imports.%import_ref.6 [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4 = fn_decl @Convert.3 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: %C.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%C [template = constants.%C.2]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.6 = fn_decl @Convert.4 [template = constants.%Convert.6] {
// CHECK:STDOUT:     %self.patt: %C.3 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.3 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%C [template = constants.%C.3]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.3 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.3 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.7 = fn_decl @Convert.5 [template = constants.%Convert.7] {
// CHECK:STDOUT:     %self.patt: %C.4 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.4 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%C [template = constants.%C.4]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.4 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.4 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.8 = fn_decl @Convert.6 [template = constants.%Convert.8] {
// CHECK:STDOUT:     %self.patt: %C.5 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.5 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.5.%C [template = constants.%C.5]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.5 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.5 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9 = fn_decl @Convert.7 [template = constants.%Convert.9] {
// CHECK:STDOUT:     %self.patt: %C.6 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.6 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.6.%C [template = constants.%C.6]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.6 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.6 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.10 = fn_decl @Convert.8 [template = constants.%Convert.10] {
// CHECK:STDOUT:     %self.patt: %C.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.7.%C [template = constants.%C.7]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.7 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.7 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.11 = fn_decl @Convert.9 [template = constants.%Convert.11] {
// CHECK:STDOUT:     %self.patt: %C.8 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.8 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8.%C [template = constants.%C.8]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.8 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.12 = fn_decl @Convert.10 [template = constants.%Convert.12] {
// CHECK:STDOUT:     %self.patt: %C.9 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.9 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.9.%C [template = constants.%C.9]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.9 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.9 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int_32.loc7_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7_18: init type = call constants.%Int(%int_32.loc7_18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_18.1: type = value_of_initializer %int.make_type_signed.loc7_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %int.make_type_signed.loc7_18, %.loc7_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16: %D.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %int_32.loc7_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7_30: init type = call constants.%Int(%int_32.loc7_30) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7_30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_signed.loc7_30, %.loc7_30.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_28: %D.elem = field_decl m, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc8_31: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc8_39: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc8_40.1: %struct_type.n.m.2 = struct_literal (%int_0.loc8_31, %int_0.loc8_39)
// CHECK:STDOUT:   %impl.elem0.loc8_40.1: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_40.1: <bound method> = bound_method %int_0.loc8_31, %impl.elem0.loc8_40.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.1: init %i32 = call %Convert.bound.loc8_40.1(%int_0.loc8_31) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_40.2: init %i32 = converted %int_0.loc8_31, %int.convert_checked.loc8_40.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_40.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc8_40.4: init %i32 = initialize_from %.loc8_40.2 to %.loc8_40.3 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc8_40.2: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_40.2: <bound method> = bound_method %int_0.loc8_39, %impl.elem0.loc8_40.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.2: init %i32 = call %Convert.bound.loc8_40.2(%int_0.loc8_39) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_40.5: init %i32 = converted %int_0.loc8_39, %int.convert_checked.loc8_40.2 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_40.6: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc8_40.7: init %i32 = initialize_from %.loc8_40.5 to %.loc8_40.6 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_40.8: init %D = class_init (%.loc8_40.4, %.loc8_40.7), %return [template = constants.%D.val]
// CHECK:STDOUT:   %.loc8_41: init %D = converted %.loc8_40.1, %.loc8_40.8 [template = constants.%D.val]
// CHECK:STDOUT:   return %.loc8_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.2) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.2]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.3]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.4]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.5]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.6]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.7]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.8]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc16: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc16
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.9]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc17: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc17
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.1) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.1
// CHECK:STDOUT:   %Self.as_type => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_1.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.2) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.2
// CHECK:STDOUT:   %Self.as_type => constants.%C.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_2.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_2.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.3) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.3
// CHECK:STDOUT:   %Self.as_type => constants.%C.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_3.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.4) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.4
// CHECK:STDOUT:   %Self.as_type => constants.%C.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_4.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_4.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.5) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5
// CHECK:STDOUT:   %Self.as_type => constants.%C.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_5.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_5.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.6) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.6
// CHECK:STDOUT:   %Self.as_type => constants.%C.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_6.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_6.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.7) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.7
// CHECK:STDOUT:   %Self.as_type => constants.%C.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_7.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_7.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.8) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.8
// CHECK:STDOUT:   %Self.as_type => constants.%C.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.n.m [template]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [template]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.3 [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%int_0.2) [template]
// CHECK:STDOUT:   %C.val.1: %C.2 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.3: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.4 [template]
// CHECK:STDOUT:   %assoc0.4: %Convert.assoc_type.3 = assoc_entity element0, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %int_1.1: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%int_1.1) [template]
// CHECK:STDOUT:   %int_2.1: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%int_2.1) [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%int_3.1) [template]
// CHECK:STDOUT:   %int_4.1: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%int_4.1) [template]
// CHECK:STDOUT:   %int_5.1: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%int_5.1) [template]
// CHECK:STDOUT:   %int_6.1: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%int_6.1) [template]
// CHECK:STDOUT:   %int_7.1: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%int_7.1) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Convert.5) [template]
// CHECK:STDOUT:   %int_1.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.2: %C.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %interface.3: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %int_2.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.3: %C.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %interface.4: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.4: %C.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %int_4.2: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_4.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.5: %C.6 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %int_5.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_5.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.6: %C.7 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.7: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %int_6.2: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_6.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.7: %C.8 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %interface.8: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %int_7.2: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_7.2, %Convert.3 [template]
// CHECK:STDOUT:   %C.val.8: %C.9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.12) [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.7
// CHECK:STDOUT:     .Make = %import_ref.43
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.10
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: type = import_ref P//library, inst+33, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref P//library, inst+47, loaded [template = constants.%complete_type.1]
// CHECK:STDOUT:   %import_ref.4 = import_ref P//library, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref P//library, inst+40, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref P//library, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.7: %C.type = import_ref P//library, inst+24, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.8: <witness> = import_ref P//library, inst+31, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.9 = import_ref P//library, inst+29, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.12: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = import_ref Core//default, inst+60, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//default, inst+71, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//default, inst+79, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.18: <witness> = import_ref Core//default, inst+103, loaded [template = constants.%interface.1]
// CHECK:STDOUT:   %import_ref.19: type = import_ref P//library, inst+143, loaded [template = constants.%C.2]
// CHECK:STDOUT:   %import_ref.20: type = import_ref P//library, inst+148, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.21: <witness> = import_ref P//library, inst+169, loaded [template = constants.%interface.2]
// CHECK:STDOUT:   %import_ref.22: type = import_ref P//library, inst+186, loaded [template = constants.%C.3]
// CHECK:STDOUT:   %import_ref.23: type = import_ref P//library, inst+191, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.24: <witness> = import_ref P//library, inst+207, loaded [template = constants.%interface.3]
// CHECK:STDOUT:   %import_ref.25: type = import_ref P//library, inst+223, loaded [template = constants.%C.4]
// CHECK:STDOUT:   %import_ref.26: type = import_ref P//library, inst+228, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.27: <witness> = import_ref P//library, inst+244, loaded [template = constants.%interface.4]
// CHECK:STDOUT:   %import_ref.28: type = import_ref P//library, inst+260, loaded [template = constants.%C.5]
// CHECK:STDOUT:   %import_ref.29: type = import_ref P//library, inst+265, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.30: <witness> = import_ref P//library, inst+281, loaded [template = constants.%interface.5]
// CHECK:STDOUT:   %import_ref.31: type = import_ref P//library, inst+297, loaded [template = constants.%C.6]
// CHECK:STDOUT:   %import_ref.32: type = import_ref P//library, inst+302, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.33: <witness> = import_ref P//library, inst+318, loaded [template = constants.%interface.6]
// CHECK:STDOUT:   %import_ref.34: type = import_ref P//library, inst+334, loaded [template = constants.%C.7]
// CHECK:STDOUT:   %import_ref.35: type = import_ref P//library, inst+339, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.36: <witness> = import_ref P//library, inst+355, loaded [template = constants.%interface.7]
// CHECK:STDOUT:   %import_ref.37: type = import_ref P//library, inst+371, loaded [template = constants.%C.8]
// CHECK:STDOUT:   %import_ref.38: type = import_ref P//library, inst+376, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.39: <witness> = import_ref P//library, inst+392, loaded [template = constants.%interface.8]
// CHECK:STDOUT:   %import_ref.40: type = import_ref P//library, inst+408, loaded [template = constants.%C.9]
// CHECK:STDOUT:   %import_ref.41: type = import_ref P//library, inst+413, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.42: <witness> = import_ref P//library, inst+429, loaded [template = constants.%interface.9]
// CHECK:STDOUT:   %import_ref.43: %Make.type = import_ref P//library, inst+54, loaded [template = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [template = constants.%F0] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_10.2: type = converted %int.make_type_signed, %.loc7_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%D]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, imports.%import_ref.14 [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.11
// CHECK:STDOUT:     .Convert = imports.%import_ref.12
// CHECK:STDOUT:     witness = (imports.%import_ref.13)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.28 as imports.%import_ref.29 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: imports.%import_ref.31 as imports.%import_ref.32 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.34 as imports.%import_ref.35 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.37 as imports.%import_ref.38 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: imports.%import_ref.40 as imports.%import_ref.41 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .n = imports.%import_ref.5
// CHECK:STDOUT:   .m = imports.%import_ref.6
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%N: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: %i32) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc8: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc8_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_34: <bound method> = bound_method %int_0, %impl.elem0.loc8_34 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.bound.loc8_34(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_0, %.loc8_34.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_0.2) [template = constants.%C.2]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.2 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.2 = class_init (), %.loc8_24.2 [template = constants.%C.val.1]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.2 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.2 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %Convert.type.4 = interface_witness_access constants.%interface.2, element0 [template = constants.%Convert.5]
// CHECK:STDOUT:   %Convert.bound.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.2 = bind_value %.loc8_26.1
// CHECK:STDOUT:   %Convert.call.loc8: init %D = call %Convert.bound.loc8_35(%.loc8_26.2) to %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.2: init %D = converted %.loc8_26.1, %Convert.call.loc8
// CHECK:STDOUT:   return %.loc8_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc9 br !if.then.loc9 else br !if.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc9:
// CHECK:STDOUT:   %.loc9_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc9: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc9_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_34: <bound method> = bound_method %int_1, %impl.elem0.loc9_34 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %Convert.bound.loc9_34(%int_1) [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %int.convert_checked.loc9 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_1, %.loc9_34.1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_1.1) [template = constants.%C.3]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.3 = class_init (), %.loc9_24.2 [template = constants.%C.val.2]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.3 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.3 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %Convert.type.4 = interface_witness_access constants.%interface.3, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %Convert.bound.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.3 = bind_value %.loc9_26.1
// CHECK:STDOUT:   %Convert.call.loc9: init %D = call %Convert.bound.loc9_35(%.loc9_26.2) to %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.2: init %D = converted %.loc9_26.1, %Convert.call.loc9
// CHECK:STDOUT:   return %.loc9_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc9:
// CHECK:STDOUT:   %false.loc10: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc10 br !if.then.loc10 else br !if.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc10:
// CHECK:STDOUT:   %.loc10_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc10: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc10_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_34: <bound method> = bound_method %int_2, %impl.elem0.loc10_34 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.bound.loc10_34(%int_2) [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_2, %.loc10_34.1 [template = constants.%int_2.1]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_2.1) [template = constants.%C.4]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.4 = class_init (), %.loc10_24.2 [template = constants.%C.val.3]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.4 = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.4 = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %Convert.type.4 = interface_witness_access constants.%interface.4, element0 [template = constants.%Convert.7]
// CHECK:STDOUT:   %Convert.bound.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.4 = bind_value %.loc10_26.1
// CHECK:STDOUT:   %Convert.call.loc10: init %D = call %Convert.bound.loc10_35(%.loc10_26.2) to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: init %D = converted %.loc10_26.1, %Convert.call.loc10
// CHECK:STDOUT:   return %.loc10_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc10:
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %.loc11_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc11: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_34: <bound method> = bound_method %int_3, %impl.elem0.loc11_34 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.bound.loc11_34(%int_3) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_3, %.loc11_34.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_3.1) [template = constants.%C.5]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.5 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.5 = class_init (), %.loc11_24.2 [template = constants.%C.val.4]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.5 = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.5 = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %Convert.type.4 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.8]
// CHECK:STDOUT:   %Convert.bound.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.5 = bind_value %.loc11_26.1
// CHECK:STDOUT:   %Convert.call.loc11: init %D = call %Convert.bound.loc11_35(%.loc11_26.2) to %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.2: init %D = converted %.loc11_26.1, %Convert.call.loc11
// CHECK:STDOUT:   return %.loc11_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc12_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc12: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.2]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12_34: <bound method> = bound_method %int_4, %impl.elem0.loc12_34 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.bound.loc12_34(%int_4) [template = constants.%int_4.1]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_4.1]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_4, %.loc12_34.1 [template = constants.%int_4.1]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_4.1) [template = constants.%C.6]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.6 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.6 = class_init (), %.loc12_24.2 [template = constants.%C.val.5]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.6 = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.6 = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %Convert.type.4 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.6 = bind_value %.loc12_26.1
// CHECK:STDOUT:   %Convert.call.loc12: init %D = call %Convert.bound.loc12_35(%.loc12_26.2) to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: init %D = converted %.loc12_26.1, %Convert.call.loc12
// CHECK:STDOUT:   return %.loc12_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc13 br !if.then.loc13 else br !if.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc13: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_34: <bound method> = bound_method %int_5, %impl.elem0.loc13_34 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.bound.loc13_34(%int_5) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_5, %.loc13_34.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_5.1) [template = constants.%C.7]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.7 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.7 = class_init (), %.loc13_24.2 [template = constants.%C.val.6]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.7 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.7 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %Convert.type.4 = interface_witness_access constants.%interface.7, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.7 = bind_value %.loc13_26.1
// CHECK:STDOUT:   %Convert.call.loc13: init %D = call %Convert.bound.loc13_35(%.loc13_26.2) to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: init %D = converted %.loc13_26.1, %Convert.call.loc13
// CHECK:STDOUT:   return %.loc13_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc13:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc14 br !if.then.loc14 else br !if.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc14:
// CHECK:STDOUT:   %.loc14_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc14: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.2]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_34: <bound method> = bound_method %int_6, %impl.elem0.loc14_34 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.bound.loc14_34(%int_6) [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_6, %.loc14_34.1 [template = constants.%int_6.1]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_6.1) [template = constants.%C.8]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.8 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.8 = class_init (), %.loc14_24.2 [template = constants.%C.val.7]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.8 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.8 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %Convert.type.4 = interface_witness_access constants.%interface.8, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %Convert.bound.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.8 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %Convert.call.loc14: init %D = call %Convert.bound.loc14_35(%.loc14_26.2) to %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.2: init %D = converted %.loc14_26.1, %Convert.call.loc14
// CHECK:STDOUT:   return %.loc14_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %false.loc15: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc15 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %.loc15_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc15: %C.type = name_ref C, imports.%import_ref.7 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.2]
// CHECK:STDOUT:   %impl.elem0.loc15_34: %Convert.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc15_34: <bound method> = bound_method %int_7, %impl.elem0.loc15_34 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.bound.loc15_34(%int_7) [template = constants.%int_7.1]
// CHECK:STDOUT:   %.loc15_34.1: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_7.1]
// CHECK:STDOUT:   %.loc15_34.2: %i32 = converted %int_7, %.loc15_34.1 [template = constants.%int_7.1]
// CHECK:STDOUT:   %C.loc15: type = class_type @C, @C(constants.%int_7.1) [template = constants.%C.9]
// CHECK:STDOUT:   %.loc15_24.2: ref %C.9 = temporary_storage
// CHECK:STDOUT:   %.loc15_24.3: init %C.9 = class_init (), %.loc15_24.2 [template = constants.%C.val.8]
// CHECK:STDOUT:   %.loc15_24.4: ref %C.9 = temporary %.loc15_24.2, %.loc15_24.3
// CHECK:STDOUT:   %.loc15_26.1: ref %C.9 = converted %.loc15_24.1, %.loc15_24.4
// CHECK:STDOUT:   %impl.elem0.loc15_35: %Convert.type.4 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.12]
// CHECK:STDOUT:   %Convert.bound.loc15_35: <bound method> = bound_method %.loc15_26.1, %impl.elem0.loc15_35
// CHECK:STDOUT:   %.loc15_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc15_26.2: %C.9 = bind_value %.loc15_26.1
// CHECK:STDOUT:   %Convert.call.loc15: init %D = call %Convert.bound.loc15_35(%.loc15_26.2) to %.loc15_35.1
// CHECK:STDOUT:   %.loc15_35.2: init %D = converted %.loc15_26.1, %Convert.call.loc15
// CHECK:STDOUT:   return %.loc15_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %P.ref.loc16: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%import_ref.43 [template = constants.%Make]
// CHECK:STDOUT:   %.loc7_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.2) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.2]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.3]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.4]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.5]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.6]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.7]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.8]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.9]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.2) {
// CHECK:STDOUT:   %N => constants.%int_0.2
// CHECK:STDOUT:   %N.patt => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.1) {
// CHECK:STDOUT:   %N => constants.%int_1.1
// CHECK:STDOUT:   %N.patt => constants.%int_1.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.1) {
// CHECK:STDOUT:   %N => constants.%int_2.1
// CHECK:STDOUT:   %N.patt => constants.%int_2.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.1) {
// CHECK:STDOUT:   %N => constants.%int_3.1
// CHECK:STDOUT:   %N.patt => constants.%int_3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.1) {
// CHECK:STDOUT:   %N => constants.%int_4.1
// CHECK:STDOUT:   %N.patt => constants.%int_4.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.1) {
// CHECK:STDOUT:   %N => constants.%int_5.1
// CHECK:STDOUT:   %N.patt => constants.%int_5.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.1) {
// CHECK:STDOUT:   %N => constants.%int_6.1
// CHECK:STDOUT:   %N.patt => constants.%int_6.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.1) {
// CHECK:STDOUT:   %N => constants.%int_7.1
// CHECK:STDOUT:   %N.patt => constants.%int_7.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
