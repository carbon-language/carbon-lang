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

fn Int32() -> type = "int.make_type_32";

fn IntLiteral() -> type = "int_literal.make_type";

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs] {
// CHECK:STDOUT:     %T.patt.loc9_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc9_22.1, runtime_param<invalid> [symbolic = %T.patt.loc9_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc9_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc9_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc13_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %int_literal.make_type, %.loc13_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_32.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc13_32.2: type = converted %int.make_type_32, %.loc13_32.1 [template = i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc9_22.1: type) {
// CHECK:STDOUT:   %T.loc9_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc9_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc9_22.2) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.loc10_32.2: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.loc10_32.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc10_32.3: @ImplicitAs.%.loc10_32.2 (%.1) = assoc_entity element0, %Convert.decl [symbolic = %.loc10_32.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self (%Self) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_20.1: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc10_20.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc10_20.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc10_20.3: type = converted %Self.ref, %.loc10_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc9_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self (%Self) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.1.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_32.1: @ImplicitAs.%.loc10_32.2 (%.1) = assoc_entity element0, %Convert.decl [symbolic = %.loc10_32.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc10_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc13_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc13_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_31.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_31.2: type = converted %int.make_type_32, %.loc14_31.1 [template = i32]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_38: <witness> = interface_witness (%Convert.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc13_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc9_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc9_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T.loc9_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T.loc9_22.2) {
// CHECK:STDOUT:   %T.loc9_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %T.loc9_22.2 => i32
// CHECK:STDOUT:   %T.patt.loc9_22.2 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.3
// CHECK:STDOUT:   %.loc10_32.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(i32, Core.IntLiteral) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %D, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.n: i32, .m: i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %.15: <bound method> = bound_method %.7, %Convert.3 [template]
// CHECK:STDOUT:   %.16: i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct: %D = struct_value (%.16, %.16) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.16) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %.20: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %.21, %Convert.3 [template]
// CHECK:STDOUT:   %.23: i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%.23) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.25, %Convert.3 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%.27) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.29, %Convert.3 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%.31) [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.33, %Convert.3 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%.35) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.36: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %.37: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.37, %Convert.3 [template]
// CHECK:STDOUT:   %.39: i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%.39) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.40: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %.41: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.41, %Convert.3 [template]
// CHECK:STDOUT:   %.43: i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%.43) [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.44: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.3 [template]
// CHECK:STDOUT:   %.47: i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.10: type = class_type @C, @C(%.47) [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.48: <witness> = interface_witness (%Convert.12) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//default, inst+5, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//default, inst+22, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.9) = import_ref Core//default, inst+50, loaded [symbolic = @ImplicitAs.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%Convert.type (%Convert.type.1) = import_ref Core//default, inst+43, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//default, inst+61, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//default, inst+66, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//default, inst+88, loaded [template = constants.%.14]
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
// CHECK:STDOUT:     %N.patt.loc6_9.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_13.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_13.2: type = converted %int.make_type_32, %.loc6_13.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_9.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N)]
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
// CHECK:STDOUT:     %.loc27_8: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:     %.loc27_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc27_7.2: <bound method> = bound_method %.loc27_8, %.loc27_7.1 [template = constants.%.15]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc27_7.2(%.loc27_8) [template = constants.%.16]
// CHECK:STDOUT:     %.loc27_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.16]
// CHECK:STDOUT:     %.loc27_7.4: i32 = converted %.loc27_8, %.loc27_7.3 [template = constants.%.16]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.16) [template = constants.%C.3]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc38_8: Core.IntLiteral = int_value 1 [template = constants.%.21]
// CHECK:STDOUT:     %.loc38_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc38_7.2: <bound method> = bound_method %.loc38_8, %.loc38_7.1 [template = constants.%.22]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc38_7.2(%.loc38_8) [template = constants.%.23]
// CHECK:STDOUT:     %.loc38_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.23]
// CHECK:STDOUT:     %.loc38_7.4: i32 = converted %.loc38_8, %.loc38_7.3 [template = constants.%.23]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.23) [template = constants.%C.4]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc49_8: Core.IntLiteral = int_value 2 [template = constants.%.25]
// CHECK:STDOUT:     %.loc49_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc49_7.2: <bound method> = bound_method %.loc49_8, %.loc49_7.1 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc49_7.2(%.loc49_8) [template = constants.%.27]
// CHECK:STDOUT:     %.loc49_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:     %.loc49_7.4: i32 = converted %.loc49_8, %.loc49_7.3 [template = constants.%.27]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.27) [template = constants.%C.5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc60_8: Core.IntLiteral = int_value 3 [template = constants.%.29]
// CHECK:STDOUT:     %.loc60_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc60_7.2: <bound method> = bound_method %.loc60_8, %.loc60_7.1 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc60_7.2(%.loc60_8) [template = constants.%.31]
// CHECK:STDOUT:     %.loc60_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:     %.loc60_7.4: i32 = converted %.loc60_8, %.loc60_7.3 [template = constants.%.31]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.31) [template = constants.%C.6]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc71_8: Core.IntLiteral = int_value 4 [template = constants.%.33]
// CHECK:STDOUT:     %.loc71_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc71_7.2: <bound method> = bound_method %.loc71_8, %.loc71_7.1 [template = constants.%.34]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc71_7.2(%.loc71_8) [template = constants.%.35]
// CHECK:STDOUT:     %.loc71_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.35]
// CHECK:STDOUT:     %.loc71_7.4: i32 = converted %.loc71_8, %.loc71_7.3 [template = constants.%.35]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.35) [template = constants.%C.7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc82_8: Core.IntLiteral = int_value 5 [template = constants.%.37]
// CHECK:STDOUT:     %.loc82_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc82_7.2: <bound method> = bound_method %.loc82_8, %.loc82_7.1 [template = constants.%.38]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc82_7.2(%.loc82_8) [template = constants.%.39]
// CHECK:STDOUT:     %.loc82_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.39]
// CHECK:STDOUT:     %.loc82_7.4: i32 = converted %.loc82_8, %.loc82_7.3 [template = constants.%.39]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.39) [template = constants.%C.8]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc93_8: Core.IntLiteral = int_value 6 [template = constants.%.41]
// CHECK:STDOUT:     %.loc93_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc93_7.2: <bound method> = bound_method %.loc93_8, %.loc93_7.1 [template = constants.%.42]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc93_7.2(%.loc93_8) [template = constants.%.43]
// CHECK:STDOUT:     %.loc93_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.43]
// CHECK:STDOUT:     %.loc93_7.4: i32 = converted %.loc93_8, %.loc93_7.3 [template = constants.%.43]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.43) [template = constants.%C.9]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc104_8: Core.IntLiteral = int_value 7 [template = constants.%.45]
// CHECK:STDOUT:     %.loc104_7.1: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:     %.loc104_7.2: <bound method> = bound_method %.loc104_8, %.loc104_7.1 [template = constants.%.46]
// CHECK:STDOUT:     %int.convert_checked: init i32 = call %.loc104_7.2(%.loc104_8) [template = constants.%.47]
// CHECK:STDOUT:     %.loc104_7.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.47]
// CHECK:STDOUT:     %.loc104_7.4: i32 = converted %.loc104_8, %.loc104_7.3 [template = constants.%.47]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.47) [template = constants.%C.10]
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.10)]
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
// CHECK:STDOUT:   %.loc27_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc27_33
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
// CHECK:STDOUT:   %.loc38_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.24]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc38_33
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
// CHECK:STDOUT:   %.loc49_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.28]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc49_33
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
// CHECK:STDOUT:   %.loc60_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc60_33
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
// CHECK:STDOUT:   %.loc71_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.36]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc71_33
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
// CHECK:STDOUT:   %.loc82_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.40]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc82_33
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
// CHECK:STDOUT:   %.loc93_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.44]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc93_33
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
// CHECK:STDOUT:   %.loc104_33: <witness> = interface_witness (%Convert.decl) [template = constants.%.48]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc104_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: i32) {
// CHECK:STDOUT:   %N.loc6_9.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6_19: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int.make_type_32.loc7_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_18.1: type = value_of_initializer %int.make_type_32.loc7_18 [template = i32]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %int.make_type_32.loc7_18, %.loc7_18.1 [template = i32]
// CHECK:STDOUT:   %.loc7_16: %.3 = field_decl n, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc7_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_32.loc7_30 [template = i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_32.loc7_30, %.loc7_30.1 [template = i32]
// CHECK:STDOUT:   %.loc7_28: %.3 = field_decl m, element1 [template]
// CHECK:STDOUT:   %.loc7_35: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_31: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_39: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_40.1: %.8 = struct_literal (%.loc15_31, %.loc15_39)
// CHECK:STDOUT:   %.loc15_40.2: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc15_40.3: <bound method> = bound_method %.loc15_31, %.loc15_40.2 [template = constants.%.15]
// CHECK:STDOUT:   %int.convert_checked.loc15_40.1: init i32 = call %.loc15_40.3(%.loc15_31) [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.4: init i32 = converted %.loc15_31, %int.convert_checked.loc15_40.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.5: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_40.6: init i32 = initialize_from %.loc15_40.4 to %.loc15_40.5 [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.7: %Convert.type.2 = interface_witness_access constants.%.14, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc15_40.8: <bound method> = bound_method %.loc15_39, %.loc15_40.7 [template = constants.%.15]
// CHECK:STDOUT:   %int.convert_checked.loc15_40.2: init i32 = call %.loc15_40.8(%.loc15_39) [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.9: init i32 = converted %.loc15_39, %int.convert_checked.loc15_40.2 [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.10: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc15_40.11: init i32 = initialize_from %.loc15_40.9 to %.loc15_40.10 [template = constants.%.16]
// CHECK:STDOUT:   %.loc15_40.12: init %D = class_init (%.loc15_40.6, %.loc15_40.11), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_41: init %D = converted %.loc15_40.1, %.loc15_40.12 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc15_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.3]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc27: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc27
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.4]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc38: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc38
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.5]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc49: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc49
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.6]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc60: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc60
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.7]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc71: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc71
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.8]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc82: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc82
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.9]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc93: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc93
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.10]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc104: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc104
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
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T) {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %T.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.16) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.16
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.16
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.3) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.23) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.23
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.23
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.4) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.27) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.27
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.27
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.5) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.31) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.31
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.31
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.6) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.35) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.35
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.35
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.7) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.39) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.39
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.8) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.43) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.43
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.43
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.9) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.47) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%.47
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%.47
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%C.10) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%C.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {.n: i32, .m: i32} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [template]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.12: %.8 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %.14: <bound method> = bound_method %.7, %Convert.3 [template]
// CHECK:STDOUT:   %.15: i32 = int_value 0 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.15) [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %struct.1: %C.3 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.4 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.19: i32 = int_value 1 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%.19) [template]
// CHECK:STDOUT:   %.20: i32 = int_value 2 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%.20) [template]
// CHECK:STDOUT:   %.21: i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.6: type = class_type @C, @C(%.21) [template]
// CHECK:STDOUT:   %.22: i32 = int_value 4 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%.22) [template]
// CHECK:STDOUT:   %.23: i32 = int_value 5 [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%.23) [template]
// CHECK:STDOUT:   %.24: i32 = int_value 6 [template]
// CHECK:STDOUT:   %C.9: type = class_type @C, @C(%.24) [template]
// CHECK:STDOUT:   %.25: i32 = int_value 7 [template]
// CHECK:STDOUT:   %C.10: type = class_type @C, @C(%.25) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.5) [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.27, %Convert.3 [template]
// CHECK:STDOUT:   %struct.2: %C.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.3 [template]
// CHECK:STDOUT:   %struct.3: %C.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.33, %Convert.3 [template]
// CHECK:STDOUT:   %struct.4: %C.6 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.36, %Convert.3 [template]
// CHECK:STDOUT:   %struct.5: %C.7 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.38: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.3 [template]
// CHECK:STDOUT:   %struct.6: %C.8 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.41: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.42, %Convert.3 [template]
// CHECK:STDOUT:   %struct.7: %C.9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.44: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.3 [template]
// CHECK:STDOUT:   %struct.8: %C.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.47: <witness> = interface_witness (%Convert.12) [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//default, inst+5, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref P//library, inst+29, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.3 = import_ref P//library, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref P//library, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref P//library, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.6: %C.type = import_ref P//library, inst+20, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.7 = import_ref P//library, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.8: %ImplicitAs.type.1 = import_ref Core//default, inst+22, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//default, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.10: @ImplicitAs.%.1 (%.8) = import_ref Core//default, inst+50, loaded [symbolic = @ImplicitAs.%.2 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//default, inst+61, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//default, inst+66, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.16: <witness> = import_ref Core//default, inst+88, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.17: type = import_ref P//library, inst+144, loaded [template = constants.%C.3]
// CHECK:STDOUT:   %import_ref.18: type = import_ref P//library, inst+149, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.19: <witness> = import_ref P//library, inst+169, loaded [template = constants.%.26]
// CHECK:STDOUT:   %import_ref.20: type = import_ref P//library, inst+186, loaded [template = constants.%C.4]
// CHECK:STDOUT:   %import_ref.21: type = import_ref P//library, inst+191, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.22: <witness> = import_ref P//library, inst+206, loaded [template = constants.%.29]
// CHECK:STDOUT:   %import_ref.23: type = import_ref P//library, inst+222, loaded [template = constants.%C.5]
// CHECK:STDOUT:   %import_ref.24: type = import_ref P//library, inst+227, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.25: <witness> = import_ref P//library, inst+242, loaded [template = constants.%.32]
// CHECK:STDOUT:   %import_ref.26: type = import_ref P//library, inst+258, loaded [template = constants.%C.6]
// CHECK:STDOUT:   %import_ref.27: type = import_ref P//library, inst+263, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.28: <witness> = import_ref P//library, inst+278, loaded [template = constants.%.35]
// CHECK:STDOUT:   %import_ref.29: type = import_ref P//library, inst+294, loaded [template = constants.%C.7]
// CHECK:STDOUT:   %import_ref.30: type = import_ref P//library, inst+299, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.31: <witness> = import_ref P//library, inst+314, loaded [template = constants.%.38]
// CHECK:STDOUT:   %import_ref.32: type = import_ref P//library, inst+330, loaded [template = constants.%C.8]
// CHECK:STDOUT:   %import_ref.33: type = import_ref P//library, inst+335, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.34: <witness> = import_ref P//library, inst+350, loaded [template = constants.%.41]
// CHECK:STDOUT:   %import_ref.35: type = import_ref P//library, inst+366, loaded [template = constants.%C.9]
// CHECK:STDOUT:   %import_ref.36: type = import_ref P//library, inst+371, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.37: <witness> = import_ref P//library, inst+386, loaded [template = constants.%.44]
// CHECK:STDOUT:   %import_ref.38: type = import_ref P//library, inst+402, loaded [template = constants.%C.10]
// CHECK:STDOUT:   %import_ref.39: type = import_ref P//library, inst+407, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.40: <witness> = import_ref P//library, inst+422, loaded [template = constants.%.47]
// CHECK:STDOUT:   %import_ref.41: %Make.type = import_ref P//library, inst+48, loaded [template = constants.%Make]
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
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc7_10.2: type = converted %int.make_type_32, %.loc7_10.1 [template = i32]
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%D]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.8) = assoc_entity element0, imports.%import_ref.12 [symbolic = %.2 (constants.%.9)]
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
// CHECK:STDOUT: generic class @C(constants.%N: i32) {
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: i32) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc19_7 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   %.loc19_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc19: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc19: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc19_33: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc19_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc19_32.2: <bound method> = bound_method %.loc19_33, %.loc19_32.1 [template = constants.%.14]
// CHECK:STDOUT:   %int.convert_checked.loc19: init i32 = call %.loc19_32.2(%.loc19_33) [template = constants.%.15]
// CHECK:STDOUT:   %.loc19_32.3: i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%.15]
// CHECK:STDOUT:   %.loc19_32.4: i32 = converted %.loc19_33, %.loc19_32.3 [template = constants.%.15]
// CHECK:STDOUT:   %C.loc19: type = class_type @C, @C(constants.%.15) [template = constants.%C.3]
// CHECK:STDOUT:   %.loc19_24.2: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %.loc19_24.3: init %C.3 = class_init (), %.loc19_24.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc19_24.4: ref %C.3 = temporary %.loc19_24.2, %.loc19_24.3
// CHECK:STDOUT:   %.loc19_26.1: ref %C.3 = converted %.loc19_24.1, %.loc19_24.4
// CHECK:STDOUT:   %.loc19_35.1: %Convert.type.4 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.5]
// CHECK:STDOUT:   %.loc19_35.2: <bound method> = bound_method %.loc19_26.1, %.loc19_35.1
// CHECK:STDOUT:   %.loc19_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc19_26.2: %C.3 = bind_value %.loc19_26.1
// CHECK:STDOUT:   %Convert.call.loc19: init %D = call %.loc19_35.2(%.loc19_26.2) to %.loc19_35.3
// CHECK:STDOUT:   %.loc19_35.4: init %D = converted %.loc19_26.1, %Convert.call.loc19
// CHECK:STDOUT:   return %.loc19_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   %.loc31_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc31_7 br !if.then.loc31 else br !if.else.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc31:
// CHECK:STDOUT:   %.loc31_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc31: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc31: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc31_33: Core.IntLiteral = int_value 1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc31_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc31_32.2: <bound method> = bound_method %.loc31_33, %.loc31_32.1 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc31: init i32 = call %.loc31_32.2(%.loc31_33) [template = constants.%.19]
// CHECK:STDOUT:   %.loc31_32.3: i32 = value_of_initializer %int.convert_checked.loc31 [template = constants.%.19]
// CHECK:STDOUT:   %.loc31_32.4: i32 = converted %.loc31_33, %.loc31_32.3 [template = constants.%.19]
// CHECK:STDOUT:   %C.loc31: type = class_type @C, @C(constants.%.19) [template = constants.%C.4]
// CHECK:STDOUT:   %.loc31_24.2: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %.loc31_24.3: init %C.4 = class_init (), %.loc31_24.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc31_24.4: ref %C.4 = temporary %.loc31_24.2, %.loc31_24.3
// CHECK:STDOUT:   %.loc31_26.1: ref %C.4 = converted %.loc31_24.1, %.loc31_24.4
// CHECK:STDOUT:   %.loc31_35.1: %Convert.type.4 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc31_35.2: <bound method> = bound_method %.loc31_26.1, %.loc31_35.1
// CHECK:STDOUT:   %.loc31_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc31_26.2: %C.4 = bind_value %.loc31_26.1
// CHECK:STDOUT:   %Convert.call.loc31: init %D = call %.loc31_35.2(%.loc31_26.2) to %.loc31_35.3
// CHECK:STDOUT:   %.loc31_35.4: init %D = converted %.loc31_26.1, %Convert.call.loc31
// CHECK:STDOUT:   return %.loc31_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc31:
// CHECK:STDOUT:   %.loc43_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc43_7 br !if.then.loc43 else br !if.else.loc43
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc43:
// CHECK:STDOUT:   %.loc43_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc43: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc43: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc43_33: Core.IntLiteral = int_value 2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc43_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc43_32.2: <bound method> = bound_method %.loc43_33, %.loc43_32.1 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc43: init i32 = call %.loc43_32.2(%.loc43_33) [template = constants.%.20]
// CHECK:STDOUT:   %.loc43_32.3: i32 = value_of_initializer %int.convert_checked.loc43 [template = constants.%.20]
// CHECK:STDOUT:   %.loc43_32.4: i32 = converted %.loc43_33, %.loc43_32.3 [template = constants.%.20]
// CHECK:STDOUT:   %C.loc43: type = class_type @C, @C(constants.%.20) [template = constants.%C.5]
// CHECK:STDOUT:   %.loc43_24.2: ref %C.5 = temporary_storage
// CHECK:STDOUT:   %.loc43_24.3: init %C.5 = class_init (), %.loc43_24.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc43_24.4: ref %C.5 = temporary %.loc43_24.2, %.loc43_24.3
// CHECK:STDOUT:   %.loc43_26.1: ref %C.5 = converted %.loc43_24.1, %.loc43_24.4
// CHECK:STDOUT:   %.loc43_35.1: %Convert.type.4 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.7]
// CHECK:STDOUT:   %.loc43_35.2: <bound method> = bound_method %.loc43_26.1, %.loc43_35.1
// CHECK:STDOUT:   %.loc43_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc43_26.2: %C.5 = bind_value %.loc43_26.1
// CHECK:STDOUT:   %Convert.call.loc43: init %D = call %.loc43_35.2(%.loc43_26.2) to %.loc43_35.3
// CHECK:STDOUT:   %.loc43_35.4: init %D = converted %.loc43_26.1, %Convert.call.loc43
// CHECK:STDOUT:   return %.loc43_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc43:
// CHECK:STDOUT:   %.loc55_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc55_7 br !if.then.loc55 else br !if.else.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc55:
// CHECK:STDOUT:   %.loc55_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc55: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc55: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc55_33: Core.IntLiteral = int_value 3 [template = constants.%.33]
// CHECK:STDOUT:   %.loc55_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc55_32.2: <bound method> = bound_method %.loc55_33, %.loc55_32.1 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc55: init i32 = call %.loc55_32.2(%.loc55_33) [template = constants.%.21]
// CHECK:STDOUT:   %.loc55_32.3: i32 = value_of_initializer %int.convert_checked.loc55 [template = constants.%.21]
// CHECK:STDOUT:   %.loc55_32.4: i32 = converted %.loc55_33, %.loc55_32.3 [template = constants.%.21]
// CHECK:STDOUT:   %C.loc55: type = class_type @C, @C(constants.%.21) [template = constants.%C.6]
// CHECK:STDOUT:   %.loc55_24.2: ref %C.6 = temporary_storage
// CHECK:STDOUT:   %.loc55_24.3: init %C.6 = class_init (), %.loc55_24.2 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc55_24.4: ref %C.6 = temporary %.loc55_24.2, %.loc55_24.3
// CHECK:STDOUT:   %.loc55_26.1: ref %C.6 = converted %.loc55_24.1, %.loc55_24.4
// CHECK:STDOUT:   %.loc55_35.1: %Convert.type.4 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.8]
// CHECK:STDOUT:   %.loc55_35.2: <bound method> = bound_method %.loc55_26.1, %.loc55_35.1
// CHECK:STDOUT:   %.loc55_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc55_26.2: %C.6 = bind_value %.loc55_26.1
// CHECK:STDOUT:   %Convert.call.loc55: init %D = call %.loc55_35.2(%.loc55_26.2) to %.loc55_35.3
// CHECK:STDOUT:   %.loc55_35.4: init %D = converted %.loc55_26.1, %Convert.call.loc55
// CHECK:STDOUT:   return %.loc55_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc55:
// CHECK:STDOUT:   %.loc67_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc67_7 br !if.then.loc67 else br !if.else.loc67
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc67:
// CHECK:STDOUT:   %.loc67_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc67: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc67: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc67_33: Core.IntLiteral = int_value 4 [template = constants.%.36]
// CHECK:STDOUT:   %.loc67_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc67_32.2: <bound method> = bound_method %.loc67_33, %.loc67_32.1 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc67: init i32 = call %.loc67_32.2(%.loc67_33) [template = constants.%.22]
// CHECK:STDOUT:   %.loc67_32.3: i32 = value_of_initializer %int.convert_checked.loc67 [template = constants.%.22]
// CHECK:STDOUT:   %.loc67_32.4: i32 = converted %.loc67_33, %.loc67_32.3 [template = constants.%.22]
// CHECK:STDOUT:   %C.loc67: type = class_type @C, @C(constants.%.22) [template = constants.%C.7]
// CHECK:STDOUT:   %.loc67_24.2: ref %C.7 = temporary_storage
// CHECK:STDOUT:   %.loc67_24.3: init %C.7 = class_init (), %.loc67_24.2 [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc67_24.4: ref %C.7 = temporary %.loc67_24.2, %.loc67_24.3
// CHECK:STDOUT:   %.loc67_26.1: ref %C.7 = converted %.loc67_24.1, %.loc67_24.4
// CHECK:STDOUT:   %.loc67_35.1: %Convert.type.4 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %.loc67_35.2: <bound method> = bound_method %.loc67_26.1, %.loc67_35.1
// CHECK:STDOUT:   %.loc67_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc67_26.2: %C.7 = bind_value %.loc67_26.1
// CHECK:STDOUT:   %Convert.call.loc67: init %D = call %.loc67_35.2(%.loc67_26.2) to %.loc67_35.3
// CHECK:STDOUT:   %.loc67_35.4: init %D = converted %.loc67_26.1, %Convert.call.loc67
// CHECK:STDOUT:   return %.loc67_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc67:
// CHECK:STDOUT:   %.loc79_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc79_7 br !if.then.loc79 else br !if.else.loc79
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc79:
// CHECK:STDOUT:   %.loc79_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc79: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc79: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc79_33: Core.IntLiteral = int_value 5 [template = constants.%.39]
// CHECK:STDOUT:   %.loc79_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc79_32.2: <bound method> = bound_method %.loc79_33, %.loc79_32.1 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc79: init i32 = call %.loc79_32.2(%.loc79_33) [template = constants.%.23]
// CHECK:STDOUT:   %.loc79_32.3: i32 = value_of_initializer %int.convert_checked.loc79 [template = constants.%.23]
// CHECK:STDOUT:   %.loc79_32.4: i32 = converted %.loc79_33, %.loc79_32.3 [template = constants.%.23]
// CHECK:STDOUT:   %C.loc79: type = class_type @C, @C(constants.%.23) [template = constants.%C.8]
// CHECK:STDOUT:   %.loc79_24.2: ref %C.8 = temporary_storage
// CHECK:STDOUT:   %.loc79_24.3: init %C.8 = class_init (), %.loc79_24.2 [template = constants.%struct.6]
// CHECK:STDOUT:   %.loc79_24.4: ref %C.8 = temporary %.loc79_24.2, %.loc79_24.3
// CHECK:STDOUT:   %.loc79_26.1: ref %C.8 = converted %.loc79_24.1, %.loc79_24.4
// CHECK:STDOUT:   %.loc79_35.1: %Convert.type.4 = interface_witness_access constants.%.41, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %.loc79_35.2: <bound method> = bound_method %.loc79_26.1, %.loc79_35.1
// CHECK:STDOUT:   %.loc79_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc79_26.2: %C.8 = bind_value %.loc79_26.1
// CHECK:STDOUT:   %Convert.call.loc79: init %D = call %.loc79_35.2(%.loc79_26.2) to %.loc79_35.3
// CHECK:STDOUT:   %.loc79_35.4: init %D = converted %.loc79_26.1, %Convert.call.loc79
// CHECK:STDOUT:   return %.loc79_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc79:
// CHECK:STDOUT:   %.loc91_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc91_7 br !if.then.loc91 else br !if.else.loc91
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc91:
// CHECK:STDOUT:   %.loc91_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc91: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc91: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc91_33: Core.IntLiteral = int_value 6 [template = constants.%.42]
// CHECK:STDOUT:   %.loc91_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc91_32.2: <bound method> = bound_method %.loc91_33, %.loc91_32.1 [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc91: init i32 = call %.loc91_32.2(%.loc91_33) [template = constants.%.24]
// CHECK:STDOUT:   %.loc91_32.3: i32 = value_of_initializer %int.convert_checked.loc91 [template = constants.%.24]
// CHECK:STDOUT:   %.loc91_32.4: i32 = converted %.loc91_33, %.loc91_32.3 [template = constants.%.24]
// CHECK:STDOUT:   %C.loc91: type = class_type @C, @C(constants.%.24) [template = constants.%C.9]
// CHECK:STDOUT:   %.loc91_24.2: ref %C.9 = temporary_storage
// CHECK:STDOUT:   %.loc91_24.3: init %C.9 = class_init (), %.loc91_24.2 [template = constants.%struct.7]
// CHECK:STDOUT:   %.loc91_24.4: ref %C.9 = temporary %.loc91_24.2, %.loc91_24.3
// CHECK:STDOUT:   %.loc91_26.1: ref %C.9 = converted %.loc91_24.1, %.loc91_24.4
// CHECK:STDOUT:   %.loc91_35.1: %Convert.type.4 = interface_witness_access constants.%.44, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %.loc91_35.2: <bound method> = bound_method %.loc91_26.1, %.loc91_35.1
// CHECK:STDOUT:   %.loc91_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc91_26.2: %C.9 = bind_value %.loc91_26.1
// CHECK:STDOUT:   %Convert.call.loc91: init %D = call %.loc91_35.2(%.loc91_26.2) to %.loc91_35.3
// CHECK:STDOUT:   %.loc91_35.4: init %D = converted %.loc91_26.1, %Convert.call.loc91
// CHECK:STDOUT:   return %.loc91_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc91:
// CHECK:STDOUT:   %.loc102_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc102_7 br !if.then.loc102 else br !if.else.loc102
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc102:
// CHECK:STDOUT:   %.loc102_24.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc102: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc102: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc102_33: Core.IntLiteral = int_value 7 [template = constants.%.45]
// CHECK:STDOUT:   %.loc102_32.1: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc102_32.2: <bound method> = bound_method %.loc102_33, %.loc102_32.1 [template = constants.%.46]
// CHECK:STDOUT:   %int.convert_checked.loc102: init i32 = call %.loc102_32.2(%.loc102_33) [template = constants.%.25]
// CHECK:STDOUT:   %.loc102_32.3: i32 = value_of_initializer %int.convert_checked.loc102 [template = constants.%.25]
// CHECK:STDOUT:   %.loc102_32.4: i32 = converted %.loc102_33, %.loc102_32.3 [template = constants.%.25]
// CHECK:STDOUT:   %C.loc102: type = class_type @C, @C(constants.%.25) [template = constants.%C.10]
// CHECK:STDOUT:   %.loc102_24.2: ref %C.10 = temporary_storage
// CHECK:STDOUT:   %.loc102_24.3: init %C.10 = class_init (), %.loc102_24.2 [template = constants.%struct.8]
// CHECK:STDOUT:   %.loc102_24.4: ref %C.10 = temporary %.loc102_24.2, %.loc102_24.3
// CHECK:STDOUT:   %.loc102_26.1: ref %C.10 = converted %.loc102_24.1, %.loc102_24.4
// CHECK:STDOUT:   %.loc102_35.1: %Convert.type.4 = interface_witness_access constants.%.47, element0 [template = constants.%Convert.12]
// CHECK:STDOUT:   %.loc102_35.2: <bound method> = bound_method %.loc102_26.1, %.loc102_35.1
// CHECK:STDOUT:   %.loc102_35.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc102_26.2: %C.10 = bind_value %.loc102_26.1
// CHECK:STDOUT:   %Convert.call.loc102: init %D = call %.loc102_35.2(%.loc102_26.2) to %.loc102_35.3
// CHECK:STDOUT:   %.loc102_35.4: init %D = converted %.loc102_26.1, %Convert.call.loc102
// CHECK:STDOUT:   return %.loc102_35.4 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc102:
// CHECK:STDOUT:   %P.ref.loc103: <namespace> = name_ref P, imports.%P [template = imports.%P]
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
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
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T) {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %T.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.15) {
// CHECK:STDOUT:   %N => constants.%.15
// CHECK:STDOUT:   %N.patt => constants.%.15
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.19) {
// CHECK:STDOUT:   %N => constants.%.19
// CHECK:STDOUT:   %N.patt => constants.%.19
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.20) {
// CHECK:STDOUT:   %N => constants.%.20
// CHECK:STDOUT:   %N.patt => constants.%.20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.21) {
// CHECK:STDOUT:   %N => constants.%.21
// CHECK:STDOUT:   %N.patt => constants.%.21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
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
