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
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %.7: %ImplicitAs.type.2 = facet_value Core.IntLiteral, Core.IntLiteral [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.2) [template]
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
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs] {
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
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %.loc12_33: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc12_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_32.2: type = converted %int.make_type_signed, %.loc12_32.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %.loc9_32.2: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.loc9_32.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc9_32.3: @ImplicitAs.%.loc9_32.2 (%.2) = assoc_entity element0, %Convert.decl [symbolic = %.loc9_32.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%.loc9_20.1 (%.1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%.loc9_20.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc9_20.3: type = facet_access_type %Self.ref [symbolic = %.loc9_20.1 (constants.%.1)]
// CHECK:STDOUT:       %.loc9_20.4: type = converted %Self.ref, %.loc9_20.3 [symbolic = %.loc9_20.1 (constants.%.1)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%.loc9_20.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.1.%.loc9_20.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_32.1: @ImplicitAs.%.loc9_32.2 (%.2) = assoc_entity element0, %Convert.decl [symbolic = %.loc9_32.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc9_32.1
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
// CHECK:STDOUT:     %.loc13_31.1: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc13_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.3: type = converted %int.make_type_signed, %.loc13_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_38: <witness> = interface_witness (%Convert.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc12_38
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
// CHECK:STDOUT:   %.loc9_20.1: type = facet_access_type %Self [symbolic = %.loc9_20.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.loc9_20.1 (%.1)]() -> @Convert.1.%T (%T);
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
// CHECK:STDOUT:   %.loc9_20.1 => constants.%.1
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
// CHECK:STDOUT:   %.loc9_32.2 => constants.%.5
// CHECK:STDOUT:   %.loc9_32.3 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32, constants.%.7) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%.7
// CHECK:STDOUT:   %.loc9_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %D, %i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.n: %i32, .m: %i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.13: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.16: %.11 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %.8, %Convert.3 [template]
// CHECK:STDOUT:   %.19: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct: %D = struct_value (%.19, %.19) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.19) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.20: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %.21: %.20 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.22: %ImplicitAs.type.2 = facet_value %C.3, %C.3 [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.25, %Convert.3 [template]
// CHECK:STDOUT:   %.27: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%.27) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.28: %ImplicitAs.type.2 = facet_value %C.4, %C.4 [symbolic]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.3 [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%.32) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.33: %ImplicitAs.type.2 = facet_value %C.5, %C.5 [symbolic]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.35, %Convert.3 [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%.37) [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.38: %ImplicitAs.type.2 = facet_value %C.6, %C.6 [symbolic]
// CHECK:STDOUT:   %.39: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.40, %Convert.3 [template]
// CHECK:STDOUT:   %.42: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%.42) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.43: %ImplicitAs.type.2 = facet_value %C.7, %C.7 [symbolic]
// CHECK:STDOUT:   %.44: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.3 [template]
// CHECK:STDOUT:   %.47: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%.47) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.48: %ImplicitAs.type.2 = facet_value %C.8, %C.8 [symbolic]
// CHECK:STDOUT:   %.49: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %.50: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.51: <bound method> = bound_method %.50, %Convert.3 [template]
// CHECK:STDOUT:   %.52: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%.52) [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.53: %ImplicitAs.type.2 = facet_value %C.9, %C.9 [symbolic]
// CHECK:STDOUT:   %.54: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %.55: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.56: <bound method> = bound_method %.55, %Convert.3 [template]
// CHECK:STDOUT:   %.57: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.10: type = class_type @C, @C(%.57) [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.58: %ImplicitAs.type.2 = facet_value %C.10, %C.10 [symbolic]
// CHECK:STDOUT:   %.59: <witness> = interface_witness (%Convert.12) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//default, inst+30, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.11) = import_ref Core//default, inst+60, loaded [symbolic = @ImplicitAs.%.2 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%Convert.type (%Convert.type.1) = import_ref Core//default, inst+52, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//default, inst+71, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//default, inst+79, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//default, inst+103, loaded [template = constants.%.17]
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc6_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_13.3: type = converted %int.make_type_signed, %.loc6_13.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc10_8: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:     %.loc10_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc10_7.2: <bound method> = bound_method %.loc10_8, %.loc10_7.1 [template = constants.%.18]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc10_7.2(%.loc10_8) [template = constants.%.19]
// CHECK:STDOUT:     %.loc10_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.19]
// CHECK:STDOUT:     %.loc10_7.4: %i32 = converted %.loc10_8, %.loc10_7.3 [template = constants.%.19]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.19) [template = constants.%C.3]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc11_8: Core.IntLiteral = int_value 1 [template = constants.%.25]
// CHECK:STDOUT:     %.loc11_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc11_7.2: <bound method> = bound_method %.loc11_8, %.loc11_7.1 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc11_7.2(%.loc11_8) [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_7.4: %i32 = converted %.loc11_8, %.loc11_7.3 [template = constants.%.27]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.27) [template = constants.%C.4]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc12_8: Core.IntLiteral = int_value 2 [template = constants.%.30]
// CHECK:STDOUT:     %.loc12_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc12_7.2: <bound method> = bound_method %.loc12_8, %.loc12_7.1 [template = constants.%.31]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc12_7.2(%.loc12_8) [template = constants.%.32]
// CHECK:STDOUT:     %.loc12_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.32]
// CHECK:STDOUT:     %.loc12_7.4: %i32 = converted %.loc12_8, %.loc12_7.3 [template = constants.%.32]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.32) [template = constants.%C.5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc13_8: Core.IntLiteral = int_value 3 [template = constants.%.35]
// CHECK:STDOUT:     %.loc13_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc13_7.2: <bound method> = bound_method %.loc13_8, %.loc13_7.1 [template = constants.%.36]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc13_7.2(%.loc13_8) [template = constants.%.37]
// CHECK:STDOUT:     %.loc13_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.37]
// CHECK:STDOUT:     %.loc13_7.4: %i32 = converted %.loc13_8, %.loc13_7.3 [template = constants.%.37]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.37) [template = constants.%C.6]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc14_8: Core.IntLiteral = int_value 4 [template = constants.%.40]
// CHECK:STDOUT:     %.loc14_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc14_7.2: <bound method> = bound_method %.loc14_8, %.loc14_7.1 [template = constants.%.41]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc14_7.2(%.loc14_8) [template = constants.%.42]
// CHECK:STDOUT:     %.loc14_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.42]
// CHECK:STDOUT:     %.loc14_7.4: %i32 = converted %.loc14_8, %.loc14_7.3 [template = constants.%.42]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.42) [template = constants.%C.7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc15_8: Core.IntLiteral = int_value 5 [template = constants.%.45]
// CHECK:STDOUT:     %.loc15_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc15_7.2: <bound method> = bound_method %.loc15_8, %.loc15_7.1 [template = constants.%.46]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc15_7.2(%.loc15_8) [template = constants.%.47]
// CHECK:STDOUT:     %.loc15_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.47]
// CHECK:STDOUT:     %.loc15_7.4: %i32 = converted %.loc15_8, %.loc15_7.3 [template = constants.%.47]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.47) [template = constants.%C.8]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc16_8: Core.IntLiteral = int_value 6 [template = constants.%.50]
// CHECK:STDOUT:     %.loc16_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc16_7.2: <bound method> = bound_method %.loc16_8, %.loc16_7.1 [template = constants.%.51]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc16_7.2(%.loc16_8) [template = constants.%.52]
// CHECK:STDOUT:     %.loc16_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.52]
// CHECK:STDOUT:     %.loc16_7.4: %i32 = converted %.loc16_8, %.loc16_7.3 [template = constants.%.52]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.52) [template = constants.%C.9]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc17_8: Core.IntLiteral = int_value 7 [template = constants.%.55]
// CHECK:STDOUT:     %.loc17_7.1: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc17_7.2: <bound method> = bound_method %.loc17_8, %.loc17_7.1 [template = constants.%.56]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc17_7.2(%.loc17_8) [template = constants.%.57]
// CHECK:STDOUT:     %.loc17_7.3: %i32 = value_of_initializer %int.convert_checked [template = constants.%.57]
// CHECK:STDOUT:     %.loc17_7.4: %i32 = converted %.loc17_8, %.loc17_7.3 [template = constants.%.57]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.57) [template = constants.%C.10]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
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
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.11) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.12)]
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
// CHECK:STDOUT:     %self.patt: %C.3 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.3 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%C [template = constants.%C.3]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.3 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.3 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.23]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc10_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.6 = fn_decl @Convert.4 [template = constants.%Convert.6] {
// CHECK:STDOUT:     %self.patt: %C.4 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.4 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%C [template = constants.%C.4]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.4 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.4 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.29]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc11_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.7 = fn_decl @Convert.5 [template = constants.%Convert.7] {
// CHECK:STDOUT:     %self.patt: %C.5 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.5 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%C [template = constants.%C.5]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.5 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.5 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.34]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc12_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.8 = fn_decl @Convert.6 [template = constants.%Convert.8] {
// CHECK:STDOUT:     %self.patt: %C.6 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.6 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.5.%C [template = constants.%C.6]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.6 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.6 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.39]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc13_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9 = fn_decl @Convert.7 [template = constants.%Convert.9] {
// CHECK:STDOUT:     %self.patt: %C.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.6.%C [template = constants.%C.7]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.7 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.7 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.44]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc14_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.10 = fn_decl @Convert.8 [template = constants.%Convert.10] {
// CHECK:STDOUT:     %self.patt: %C.8 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.8 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.7.%C [template = constants.%C.8]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.8 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.49]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc15_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.11 = fn_decl @Convert.9 [template = constants.%Convert.11] {
// CHECK:STDOUT:     %self.patt: %C.9 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.9 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8.%C [template = constants.%C.9]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.9 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.9 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.54]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc16_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.12 = fn_decl @Convert.10 [template = constants.%Convert.12] {
// CHECK:STDOUT:     %self.patt: %C.10 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.10 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.9.%C [template = constants.%C.10]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.10 = value_param runtime_param0
// CHECK:STDOUT:     %self: %C.10 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.59]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc17_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6_19: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc7_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7_18: init type = call constants.%Int(%.loc7_18.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_18.2: type = value_of_initializer %int.make_type_signed.loc7_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_18.3: type = converted %int.make_type_signed.loc7_18, %.loc7_18.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16: %.4 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc7_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7_30: init type = call constants.%Int(%.loc7_30.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.2: type = value_of_initializer %int.make_type_signed.loc7_30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.3: type = converted %int.make_type_signed.loc7_30, %.loc7_30.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_28: %.4 = field_decl m, element1 [template]
// CHECK:STDOUT:   %.loc7_35: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_31: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_39: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_40.1: %.9 = struct_literal (%.loc8_31, %.loc8_39)
// CHECK:STDOUT:   %.loc8_40.2: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc8_40.3: <bound method> = bound_method %.loc8_31, %.loc8_40.2 [template = constants.%.18]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.1: init %i32 = call %.loc8_40.3(%.loc8_31) [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.4: init %i32 = converted %.loc8_31, %int.convert_checked.loc8_40.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.5: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc8_40.6: init %i32 = initialize_from %.loc8_40.4 to %.loc8_40.5 [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.7: %Convert.type.2 = interface_witness_access constants.%.17, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc8_40.8: <bound method> = bound_method %.loc8_39, %.loc8_40.7 [template = constants.%.18]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.2: init %i32 = call %.loc8_40.8(%.loc8_39) [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.9: init %i32 = converted %.loc8_39, %int.convert_checked.loc8_40.2 [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.10: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc8_40.11: init %i32 = initialize_from %.loc8_40.9 to %.loc8_40.10 [template = constants.%.19]
// CHECK:STDOUT:   %.loc8_40.12: init %D = class_init (%.loc8_40.6, %.loc8_40.11), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_41: init %D = converted %.loc8_40.1, %.loc8_40.12 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc8_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.10)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.3]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.4]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.5]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.6]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.7]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.8]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.9]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc16: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc16
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.10]() -> %return: %D {
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
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.19) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.19
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.19
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
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %.2 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.22) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.22
// CHECK:STDOUT:   %.1 => constants.%C.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.27) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.27
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.27
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.28) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.28
// CHECK:STDOUT:   %.1 => constants.%C.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.32) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.32
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.33) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.33
// CHECK:STDOUT:   %.1 => constants.%C.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.37) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.37
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.37
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.38) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.38
// CHECK:STDOUT:   %.1 => constants.%C.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.42) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.42
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.43) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.43
// CHECK:STDOUT:   %.1 => constants.%C.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.47) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.47
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.47
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.48) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.48
// CHECK:STDOUT:   %.1 => constants.%C.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.52) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.52
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.52
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.53) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.53
// CHECK:STDOUT:   %.1 => constants.%C.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.57) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.57
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.57
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%.58) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.58
// CHECK:STDOUT:   %.1 => constants.%C.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [template]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [template]
// CHECK:STDOUT:   %.5: bool = bool_literal false [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %.12: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.15: %.10 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %.17: <bound method> = bound_method %.8, %Convert.3 [template]
// CHECK:STDOUT:   %.18: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.18) [template]
// CHECK:STDOUT:   %struct.1: %C.3 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.20: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.4 [template]
// CHECK:STDOUT:   %.21: %.20 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.22: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%.22) [template]
// CHECK:STDOUT:   %.23: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%.23) [template]
// CHECK:STDOUT:   %.24: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%.24) [template]
// CHECK:STDOUT:   %.25: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%.25) [template]
// CHECK:STDOUT:   %.26: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%.26) [template]
// CHECK:STDOUT:   %.27: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%.27) [template]
// CHECK:STDOUT:   %.28: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.10: type = class_type @C, @C(%.28) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.5) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.3 [template]
// CHECK:STDOUT:   %struct.2: %C.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.33, %Convert.3 [template]
// CHECK:STDOUT:   %struct.3: %C.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.36, %Convert.3 [template]
// CHECK:STDOUT:   %struct.4: %C.6 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.38: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.3 [template]
// CHECK:STDOUT:   %struct.5: %C.7 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.41: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.42, %Convert.3 [template]
// CHECK:STDOUT:   %struct.6: %C.8 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.44: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.3 [template]
// CHECK:STDOUT:   %struct.7: %C.9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.47: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %.48: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.49: <bound method> = bound_method %.48, %Convert.3 [template]
// CHECK:STDOUT:   %struct.8: %C.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.50: <witness> = interface_witness (%Convert.12) [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.6
// CHECK:STDOUT:     .Make = %import_ref.41
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: type = import_ref P//library, inst+36, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.3 = import_ref P//library, inst+37, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref P//library, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref P//library, inst+48, unloaded
// CHECK:STDOUT:   %import_ref.6: %C.type = import_ref P//library, inst+27, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.7 = import_ref P//library, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.10: @ImplicitAs.%.1 (%.10) = import_ref Core//default, inst+60, loaded [symbolic = @ImplicitAs.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//default, inst+71, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//default, inst+79, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.16: <witness> = import_ref Core//default, inst+103, loaded [template = constants.%.16]
// CHECK:STDOUT:   %import_ref.17: type = import_ref P//library, inst+156, loaded [template = constants.%C.3]
// CHECK:STDOUT:   %import_ref.18: type = import_ref P//library, inst+161, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.19: <witness> = import_ref P//library, inst+182, loaded [template = constants.%.29]
// CHECK:STDOUT:   %import_ref.20: type = import_ref P//library, inst+199, loaded [template = constants.%C.4]
// CHECK:STDOUT:   %import_ref.21: type = import_ref P//library, inst+204, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.22: <witness> = import_ref P//library, inst+220, loaded [template = constants.%.32]
// CHECK:STDOUT:   %import_ref.23: type = import_ref P//library, inst+236, loaded [template = constants.%C.5]
// CHECK:STDOUT:   %import_ref.24: type = import_ref P//library, inst+241, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.25: <witness> = import_ref P//library, inst+257, loaded [template = constants.%.35]
// CHECK:STDOUT:   %import_ref.26: type = import_ref P//library, inst+273, loaded [template = constants.%C.6]
// CHECK:STDOUT:   %import_ref.27: type = import_ref P//library, inst+278, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.28: <witness> = import_ref P//library, inst+294, loaded [template = constants.%.38]
// CHECK:STDOUT:   %import_ref.29: type = import_ref P//library, inst+310, loaded [template = constants.%C.7]
// CHECK:STDOUT:   %import_ref.30: type = import_ref P//library, inst+315, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.31: <witness> = import_ref P//library, inst+331, loaded [template = constants.%.41]
// CHECK:STDOUT:   %import_ref.32: type = import_ref P//library, inst+347, loaded [template = constants.%C.8]
// CHECK:STDOUT:   %import_ref.33: type = import_ref P//library, inst+352, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.34: <witness> = import_ref P//library, inst+368, loaded [template = constants.%.44]
// CHECK:STDOUT:   %import_ref.35: type = import_ref P//library, inst+384, loaded [template = constants.%C.9]
// CHECK:STDOUT:   %import_ref.36: type = import_ref P//library, inst+389, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.37: <witness> = import_ref P//library, inst+405, loaded [template = constants.%.47]
// CHECK:STDOUT:   %import_ref.38: type = import_ref P//library, inst+421, loaded [template = constants.%C.10]
// CHECK:STDOUT:   %import_ref.39: type = import_ref P//library, inst+426, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.40: <witness> = import_ref P//library, inst+442, loaded [template = constants.%.50]
// CHECK:STDOUT:   %import_ref.41: %Make.type = import_ref P//library, inst+57, loaded [template = constants.%Make]
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
// CHECK:STDOUT:     %.loc7_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc7_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_10.3: type = converted %int.make_type_signed, %.loc7_10.2 [template = constants.%i32]
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
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.10) = assoc_entity element0, imports.%import_ref.12 [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9
// CHECK:STDOUT:     .Convert = imports.%import_ref.10
// CHECK:STDOUT:     witness = (imports.%import_ref.11)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.20 as imports.%import_ref.21 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.23 as imports.%import_ref.24 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.26 as imports.%import_ref.27 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .n = imports.%import_ref.4
// CHECK:STDOUT:   .m = imports.%import_ref.5
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
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: %i32) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc8_7 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc8_33: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc8_32.2: <bound method> = bound_method %.loc8_33, %.loc8_32.1 [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %.loc8_32.2(%.loc8_33) [template = constants.%.18]
// CHECK:STDOUT:   %.loc8_32.3: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%.18]
// CHECK:STDOUT:   %.loc8_32.4: %i32 = converted %.loc8_33, %.loc8_32.3 [template = constants.%.18]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%.18) [template = constants.%C.3]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.3 = class_init (), %.loc8_24.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.3 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.3 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %.loc8_35.1: %Convert.type.4 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.5]
// CHECK:STDOUT:   %.loc8_35.2: <bound method> = bound_method %.loc8_26.1, %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.3 = bind_value %.loc8_26.1
// CHECK:STDOUT:   %Convert.call.loc8: init %D = call %.loc8_35.2(%.loc8_26.2) to %.loc8_35.3
// CHECK:STDOUT:   %.loc8_35.4: init %D = converted %.loc8_26.1, %Convert.call.loc8
// CHECK:STDOUT:   return %.loc8_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %.loc9_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc9_7 br !if.then.loc9 else br !if.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc9:
// CHECK:STDOUT:   %.loc9_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc9: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc9_33: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc9_32.2: <bound method> = bound_method %.loc9_33, %.loc9_32.1 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %.loc9_32.2(%.loc9_33) [template = constants.%.22]
// CHECK:STDOUT:   %.loc9_32.3: %i32 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.22]
// CHECK:STDOUT:   %.loc9_32.4: %i32 = converted %.loc9_33, %.loc9_32.3 [template = constants.%.22]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%.22) [template = constants.%C.4]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.4 = class_init (), %.loc9_24.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.4 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.4 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %.loc9_35.1: %Convert.type.4 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc9_35.2: <bound method> = bound_method %.loc9_26.1, %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.4 = bind_value %.loc9_26.1
// CHECK:STDOUT:   %Convert.call.loc9: init %D = call %.loc9_35.2(%.loc9_26.2) to %.loc9_35.3
// CHECK:STDOUT:   %.loc9_35.4: init %D = converted %.loc9_26.1, %Convert.call.loc9
// CHECK:STDOUT:   return %.loc9_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc9:
// CHECK:STDOUT:   %.loc10_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc10_7 br !if.then.loc10 else br !if.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc10:
// CHECK:STDOUT:   %.loc10_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc10: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc10_33: Core.IntLiteral = int_value 2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc10_32.2: <bound method> = bound_method %.loc10_33, %.loc10_32.1 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %.loc10_32.2(%.loc10_33) [template = constants.%.23]
// CHECK:STDOUT:   %.loc10_32.3: %i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%.23]
// CHECK:STDOUT:   %.loc10_32.4: %i32 = converted %.loc10_33, %.loc10_32.3 [template = constants.%.23]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%.23) [template = constants.%C.5]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.5 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.5 = class_init (), %.loc10_24.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.5 = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.5 = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %.loc10_35.1: %Convert.type.4 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.7]
// CHECK:STDOUT:   %.loc10_35.2: <bound method> = bound_method %.loc10_26.1, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.5 = bind_value %.loc10_26.1
// CHECK:STDOUT:   %Convert.call.loc10: init %D = call %.loc10_35.2(%.loc10_26.2) to %.loc10_35.3
// CHECK:STDOUT:   %.loc10_35.4: init %D = converted %.loc10_26.1, %Convert.call.loc10
// CHECK:STDOUT:   return %.loc10_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc10:
// CHECK:STDOUT:   %.loc11_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc11_7 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %.loc11_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc11: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc11_33: Core.IntLiteral = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc11_32.2: <bound method> = bound_method %.loc11_33, %.loc11_32.1 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %.loc11_32.2(%.loc11_33) [template = constants.%.24]
// CHECK:STDOUT:   %.loc11_32.3: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%.24]
// CHECK:STDOUT:   %.loc11_32.4: %i32 = converted %.loc11_33, %.loc11_32.3 [template = constants.%.24]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%.24) [template = constants.%C.6]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.6 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.6 = class_init (), %.loc11_24.2 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.6 = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.6 = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %.loc11_35.1: %Convert.type.4 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.8]
// CHECK:STDOUT:   %.loc11_35.2: <bound method> = bound_method %.loc11_26.1, %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.6 = bind_value %.loc11_26.1
// CHECK:STDOUT:   %Convert.call.loc11: init %D = call %.loc11_35.2(%.loc11_26.2) to %.loc11_35.3
// CHECK:STDOUT:   %.loc11_35.4: init %D = converted %.loc11_26.1, %Convert.call.loc11
// CHECK:STDOUT:   return %.loc11_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %.loc12_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc12_7 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc12_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc12: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc12_33: Core.IntLiteral = int_value 4 [template = constants.%.39]
// CHECK:STDOUT:   %.loc12_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc12_32.2: <bound method> = bound_method %.loc12_33, %.loc12_32.1 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_32.2(%.loc12_33) [template = constants.%.25]
// CHECK:STDOUT:   %.loc12_32.3: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%.25]
// CHECK:STDOUT:   %.loc12_32.4: %i32 = converted %.loc12_33, %.loc12_32.3 [template = constants.%.25]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%.25) [template = constants.%C.7]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.7 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.7 = class_init (), %.loc12_24.2 [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.7 = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.7 = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %.loc12_35.1: %Convert.type.4 = interface_witness_access constants.%.41, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %.loc12_35.2: <bound method> = bound_method %.loc12_26.1, %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.7 = bind_value %.loc12_26.1
// CHECK:STDOUT:   %Convert.call.loc12: init %D = call %.loc12_35.2(%.loc12_26.2) to %.loc12_35.3
// CHECK:STDOUT:   %.loc12_35.4: init %D = converted %.loc12_26.1, %Convert.call.loc12
// CHECK:STDOUT:   return %.loc12_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %.loc13_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc13_7 br !if.then.loc13 else br !if.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc13: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc13_33: Core.IntLiteral = int_value 5 [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc13_32.2: <bound method> = bound_method %.loc13_33, %.loc13_32.1 [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %.loc13_32.2(%.loc13_33) [template = constants.%.26]
// CHECK:STDOUT:   %.loc13_32.3: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%.26]
// CHECK:STDOUT:   %.loc13_32.4: %i32 = converted %.loc13_33, %.loc13_32.3 [template = constants.%.26]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%.26) [template = constants.%C.8]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.8 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.8 = class_init (), %.loc13_24.2 [template = constants.%struct.6]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.8 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.8 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %.loc13_35.1: %Convert.type.4 = interface_witness_access constants.%.44, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %.loc13_35.2: <bound method> = bound_method %.loc13_26.1, %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.8 = bind_value %.loc13_26.1
// CHECK:STDOUT:   %Convert.call.loc13: init %D = call %.loc13_35.2(%.loc13_26.2) to %.loc13_35.3
// CHECK:STDOUT:   %.loc13_35.4: init %D = converted %.loc13_26.1, %Convert.call.loc13
// CHECK:STDOUT:   return %.loc13_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc13:
// CHECK:STDOUT:   %.loc14_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc14_7 br !if.then.loc14 else br !if.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc14:
// CHECK:STDOUT:   %.loc14_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc14: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc14_33: Core.IntLiteral = int_value 6 [template = constants.%.45]
// CHECK:STDOUT:   %.loc14_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc14_32.2: <bound method> = bound_method %.loc14_33, %.loc14_32.1 [template = constants.%.46]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %.loc14_32.2(%.loc14_33) [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_32.3: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_32.4: %i32 = converted %.loc14_33, %.loc14_32.3 [template = constants.%.27]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%.27) [template = constants.%C.9]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.9 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.9 = class_init (), %.loc14_24.2 [template = constants.%struct.7]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.9 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.9 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %.loc14_35.1: %Convert.type.4 = interface_witness_access constants.%.47, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %.loc14_35.2: <bound method> = bound_method %.loc14_26.1, %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.9 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %Convert.call.loc14: init %D = call %.loc14_35.2(%.loc14_26.2) to %.loc14_35.3
// CHECK:STDOUT:   %.loc14_35.4: init %D = converted %.loc14_26.1, %Convert.call.loc14
// CHECK:STDOUT:   return %.loc14_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %.loc15_7: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc15_7 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %.loc15_24.1: %.6 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc15: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc15_33: Core.IntLiteral = int_value 7 [template = constants.%.48]
// CHECK:STDOUT:   %.loc15_32.1: %Convert.type.2 = interface_witness_access constants.%.16, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc15_32.2: <bound method> = bound_method %.loc15_33, %.loc15_32.1 [template = constants.%.49]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %.loc15_32.2(%.loc15_33) [template = constants.%.28]
// CHECK:STDOUT:   %.loc15_32.3: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%.28]
// CHECK:STDOUT:   %.loc15_32.4: %i32 = converted %.loc15_33, %.loc15_32.3 [template = constants.%.28]
// CHECK:STDOUT:   %C.loc15: type = class_type @C, @C(constants.%.28) [template = constants.%C.10]
// CHECK:STDOUT:   %.loc15_24.2: ref %C.10 = temporary_storage
// CHECK:STDOUT:   %.loc15_24.3: init %C.10 = class_init (), %.loc15_24.2 [template = constants.%struct.8]
// CHECK:STDOUT:   %.loc15_24.4: ref %C.10 = temporary %.loc15_24.2, %.loc15_24.3
// CHECK:STDOUT:   %.loc15_26.1: ref %C.10 = converted %.loc15_24.1, %.loc15_24.4
// CHECK:STDOUT:   %.loc15_35.1: %Convert.type.4 = interface_witness_access constants.%.50, element0 [template = constants.%Convert.12]
// CHECK:STDOUT:   %.loc15_35.2: <bound method> = bound_method %.loc15_26.1, %.loc15_35.1
// CHECK:STDOUT:   %.loc15_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc15_26.2: %C.10 = bind_value %.loc15_26.1
// CHECK:STDOUT:   %Convert.call.loc15: init %D = call %.loc15_35.2(%.loc15_26.2) to %.loc15_35.3
// CHECK:STDOUT:   %.loc15_35.4: init %D = converted %.loc15_26.1, %Convert.call.loc15
// CHECK:STDOUT:   return %.loc15_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %P.ref.loc16: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%import_ref.41 [template = constants.%Make]
// CHECK:STDOUT:   %.loc7_15.2: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15.2
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.9)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.3]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.4]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.5]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.6]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.7]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.8]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.9]() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.10]() -> %D;
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
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %.2 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.18) {
// CHECK:STDOUT:   %N => constants.%.18
// CHECK:STDOUT:   %N.patt => constants.%.18
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
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %.2 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.22) {
// CHECK:STDOUT:   %N => constants.%.22
// CHECK:STDOUT:   %N.patt => constants.%.22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.23) {
// CHECK:STDOUT:   %N => constants.%.23
// CHECK:STDOUT:   %N.patt => constants.%.23
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.24) {
// CHECK:STDOUT:   %N => constants.%.24
// CHECK:STDOUT:   %N.patt => constants.%.24
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.25) {
// CHECK:STDOUT:   %N => constants.%.25
// CHECK:STDOUT:   %N.patt => constants.%.25
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.26) {
// CHECK:STDOUT:   %N => constants.%.26
// CHECK:STDOUT:   %N.patt => constants.%.26
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.27) {
// CHECK:STDOUT:   %N => constants.%.27
// CHECK:STDOUT:   %N.patt => constants.%.27
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.28) {
// CHECK:STDOUT:   %N => constants.%.28
// CHECK:STDOUT:   %N.patt => constants.%.28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
