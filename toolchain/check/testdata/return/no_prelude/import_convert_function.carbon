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
// CHECK:STDOUT:   %ImplicitAs.type.96: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.a: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type.b: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.b = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.4: type = assoc_entity_type %ImplicitAs.type.a, %Convert.type.b [symbolic]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.4 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %ImplicitAs.type.98: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.c: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.c = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.9: type = assoc_entity_type %ImplicitAs.type.98, %Convert.type.c [template]
// CHECK:STDOUT:   %assoc0.e: %Convert.assoc_type.9 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a = facet_value Core.IntLiteral, %impl_witness [symbolic]
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
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.3: type = converted %int_literal.make_type, %.loc6_22.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96 = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
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
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %int.make_type_signed, %.loc12_36.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [template = constants.%ImplicitAs.type.98]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc8_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.a)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.a = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc8_22.2) [symbolic = %Convert.type (constants.%Convert.type.b)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.b) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a), @ImplicitAs.%Convert.type (%Convert.type.b) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.4)]
// CHECK:STDOUT:   %assoc0.loc9_32.2: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.b) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.6)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %.loc9_20.1: type = splice_block %.loc9_20.3 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc9_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.a) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.a) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc9_20.3: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_32.1: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc9_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9 = fn_decl @Convert.2 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed, %.loc13_31.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc12_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc8_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.a)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.a = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc8_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.98
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.9
// CHECK:STDOUT:   %assoc0.loc9_32.2 => constants.%assoc0.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.98
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
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.8: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32.builtin [template]
// CHECK:STDOUT:   %struct_type.n.m.5: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [template]
// CHECK:STDOUT:   %complete_type.6: <witness> = complete_type_witness %struct_type.n.m.5 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct_type.n.m.8: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.c: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.c = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.60: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.de: %Convert.type.60 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.4: type = assoc_entity_type %ImplicitAs.type.d, %Convert.type.60 [symbolic]
// CHECK:STDOUT:   %assoc0.317cac.1: %Convert.assoc_type.4 = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.2a: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.75: %Convert.type.2a = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.f: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.2a [template]
// CHECK:STDOUT:   %assoc0.6: %Convert.assoc_type.f = assoc_entity element0, imports.%import_ref.dbaf9d.1 [template]
// CHECK:STDOUT:   %assoc0.317cac.2: %Convert.assoc_type.4 = assoc_entity element0, imports.%import_ref.dbaf9d.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (imports.%import_ref.02) [template]
// CHECK:STDOUT:   %Convert.type.c: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.71: %Convert.type.c = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.59: <bound method> = bound_method %int_0.5, %Convert.71 [template]
// CHECK:STDOUT:   %int_0.a: %i32.builtin = int_value 0 [template]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.a, %int_0.a) [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%int_0.a) [template]
// CHECK:STDOUT:   %ImplicitAs.type.a: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.b4: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.b: type = assoc_entity_type %ImplicitAs.type.a, %Convert.type.8 [template]
// CHECK:STDOUT:   %assoc0.7: %Convert.assoc_type.b = assoc_entity element0, imports.%import_ref.dbaf9d.1 [template]
// CHECK:STDOUT:   %impl_witness.a1: <witness> = impl_witness (@impl.2.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.e: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.b2: %Convert.type.e = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.5d: %ImplicitAs.type.d = facet_value %C.5, %impl_witness.a1 [symbolic]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.5f: <bound method> = bound_method %int_1.5, %Convert.71 [template]
// CHECK:STDOUT:   %int_1.f: %i32.builtin = int_value 1 [template]
// CHECK:STDOUT:   %C.c: type = class_type @C, @C(%int_1.f) [template]
// CHECK:STDOUT:   %impl_witness.d2: <witness> = impl_witness (@impl.3.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.62: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.0: %Convert.type.62 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.4: %ImplicitAs.type.d = facet_value %C.c, %impl_witness.d2 [symbolic]
// CHECK:STDOUT:   %int_2.e: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.a: <bound method> = bound_method %int_2.e, %Convert.71 [template]
// CHECK:STDOUT:   %int_2.5: %i32.builtin = int_value 2 [template]
// CHECK:STDOUT:   %C.75: type = class_type @C, @C(%int_2.5) [template]
// CHECK:STDOUT:   %impl_witness.de: <witness> = impl_witness (@impl.4.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.d: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.d = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.0b: %ImplicitAs.type.d = facet_value %C.75, %impl_witness.de [symbolic]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.22: <bound method> = bound_method %int_3.1, %Convert.71 [template]
// CHECK:STDOUT:   %int_3.a: %i32.builtin = int_value 3 [template]
// CHECK:STDOUT:   %C.de: type = class_type @C, @C(%int_3.a) [template]
// CHECK:STDOUT:   %impl_witness.fa: <witness> = impl_witness (@impl.5.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.b7: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.f: %ImplicitAs.type.d = facet_value %C.de, %impl_witness.fa [symbolic]
// CHECK:STDOUT:   %int_4.0: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.50: <bound method> = bound_method %int_4.0, %Convert.71 [template]
// CHECK:STDOUT:   %int_4.4: %i32.builtin = int_value 4 [template]
// CHECK:STDOUT:   %C.71: type = class_type @C, @C(%int_4.4) [template]
// CHECK:STDOUT:   %impl_witness.f4: <witness> = impl_witness (@impl.6.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.b: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.7c: %Convert.type.b = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.06: %ImplicitAs.type.d = facet_value %C.71, %impl_witness.f4 [symbolic]
// CHECK:STDOUT:   %int_5.6: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.2c: <bound method> = bound_method %int_5.6, %Convert.71 [template]
// CHECK:STDOUT:   %int_5.9: %i32.builtin = int_value 5 [template]
// CHECK:STDOUT:   %C.a: type = class_type @C, @C(%int_5.9) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (@impl.7.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.91: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.d4: %Convert.type.91 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.6: %ImplicitAs.type.d = facet_value %C.a, %impl_witness.5 [symbolic]
// CHECK:STDOUT:   %int_6.4: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.d: <bound method> = bound_method %int_6.4, %Convert.71 [template]
// CHECK:STDOUT:   %int_6.e: %i32.builtin = int_value 6 [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%int_6.e) [template]
// CHECK:STDOUT:   %impl_witness.a2: <witness> = impl_witness (@impl.8.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.25: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.f02: %Convert.type.25 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.5b: %ImplicitAs.type.d = facet_value %C.1, %impl_witness.a2 [symbolic]
// CHECK:STDOUT:   %int_7.2: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.e: <bound method> = bound_method %int_7.2, %Convert.71 [template]
// CHECK:STDOUT:   %int_7.6: %i32.builtin = int_value 7 [template]
// CHECK:STDOUT:   %C.d0: type = class_type @C, @C(%int_7.6) [template]
// CHECK:STDOUT:   %impl_witness.ac: <witness> = impl_witness (@impl.9.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.9e: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.f0d: %Convert.type.9e = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.2: %ImplicitAs.type.d = facet_value %C.d0, %impl_witness.ac [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.71
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7f
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7f: %ImplicitAs.type.c = import_ref Core//default, ImplicitAs, loaded [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %import_ref.01 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.317cac.2)]
// CHECK:STDOUT:   %import_ref.6: @ImplicitAs.%Convert.type (%Convert.type.60) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.de)]
// CHECK:STDOUT:   %import_ref.dbaf9d.1 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %import_ref.4: <witness> = import_ref Core//default, loc12_38, loaded [template = constants.%impl_witness.2]
// CHECK:STDOUT:   %import_ref.a: type = import_ref Core//default, loc12_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.d9: type = import_ref Core//default, loc12_36, loaded [template = constants.%ImplicitAs.type.5]
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
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %i32.builtin = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32.builtin = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.3 [template = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.3: type = converted %int.make_type_signed, %.loc6_13.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32.builtin = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N)]
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
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.59]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_0) [template = constants.%int_0.a]
// CHECK:STDOUT:     %.loc10_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_0.a]
// CHECK:STDOUT:     %.loc10_9.2: %i32.builtin = converted %int_0, %.loc10_9.1 [template = constants.%int_0.a]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.a) [template = constants.%C.5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (@impl.2.%Convert.decl) [template = constants.%impl_witness.a1]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.5f]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_1) [template = constants.%int_1.f]
// CHECK:STDOUT:     %.loc11_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_1.f]
// CHECK:STDOUT:     %.loc11_9.2: %i32.builtin = converted %int_1, %.loc11_9.1 [template = constants.%int_1.f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.f) [template = constants.%C.c]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (@impl.3.%Convert.decl) [template = constants.%impl_witness.d2]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.e]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound.a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_2) [template = constants.%int_2.5]
// CHECK:STDOUT:     %.loc12_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_2.5]
// CHECK:STDOUT:     %.loc12_9.2: %i32.builtin = converted %int_2, %.loc12_9.1 [template = constants.%int_2.5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.5) [template = constants.%C.75]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (@impl.4.%Convert.decl) [template = constants.%impl_witness.de]
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_3, %impl.elem0 [template = constants.%Convert.bound.22]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_3) [template = constants.%int_3.a]
// CHECK:STDOUT:     %.loc13_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_3.a]
// CHECK:STDOUT:     %.loc13_9.2: %i32.builtin = converted %int_3, %.loc13_9.1 [template = constants.%int_3.a]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.a) [template = constants.%C.de]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.5.%Convert.decl) [template = constants.%impl_witness.fa]
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_4, %impl.elem0 [template = constants.%Convert.bound.50]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_4) [template = constants.%int_4.4]
// CHECK:STDOUT:     %.loc14_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_4.4]
// CHECK:STDOUT:     %.loc14_9.2: %i32.builtin = converted %int_4, %.loc14_9.1 [template = constants.%int_4.4]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.4) [template = constants.%C.71]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (@impl.6.%Convert.decl) [template = constants.%impl_witness.f4]
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.6]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound.2c]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_5) [template = constants.%int_5.9]
// CHECK:STDOUT:     %.loc15_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_5.9]
// CHECK:STDOUT:     %.loc15_9.2: %i32.builtin = converted %int_5, %.loc15_9.1 [template = constants.%int_5.9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.9) [template = constants.%C.a]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.7.%Convert.decl) [template = constants.%impl_witness.5]
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.4]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_6, %impl.elem0 [template = constants.%Convert.bound.d]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_6) [template = constants.%int_6.e]
// CHECK:STDOUT:     %.loc16_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_6.e]
// CHECK:STDOUT:     %.loc16_9.2: %i32.builtin = converted %int_6, %.loc16_9.1 [template = constants.%int_6.e]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e) [template = constants.%C.1]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.8.%Convert.decl) [template = constants.%impl_witness.a2]
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.2]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_7, %impl.elem0 [template = constants.%Convert.bound.e]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %Convert.bound(%int_7) [template = constants.%int_7.6]
// CHECK:STDOUT:     %.loc17_9.1: %i32.builtin = value_of_initializer %int.convert_checked [template = constants.%int_7.6]
// CHECK:STDOUT:     %.loc17_9.2: %i32.builtin = converted %int_7, %.loc17_9.1 [template = constants.%int_7.6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.6) [template = constants.%C.d0]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.c = name_ref ImplicitAs, imports.%import_ref.7f [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (@impl.9.%Convert.decl) [template = constants.%impl_witness.ac]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.60)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.60) = struct_value () [symbolic = %Convert (constants.%Convert.de)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d), @ImplicitAs.%Convert.type (%Convert.type.60) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.4)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic = %assoc0 (constants.%assoc0.317cac.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.01
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.6)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.a as imports.%import_ref.d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e = fn_decl @Convert.3 [template = constants.%Convert.b2] {
// CHECK:STDOUT:     %self.patt: %C.5 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.5 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.5 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%C [template = constants.%C.5]
// CHECK:STDOUT:     %self: %C.5 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.62 = fn_decl @Convert.4 [template = constants.%Convert.0] {
// CHECK:STDOUT:     %self.patt: %C.c = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.c = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%C [template = constants.%C.c]
// CHECK:STDOUT:     %self: %C.c = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.d = fn_decl @Convert.5 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: %C.75 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.75 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.75 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%C [template = constants.%C.75]
// CHECK:STDOUT:     %self: %C.75 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.3 = fn_decl @Convert.6 [template = constants.%Convert.b7] {
// CHECK:STDOUT:     %self.patt: %C.de = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.de = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.de = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.5.%C [template = constants.%C.de]
// CHECK:STDOUT:     %self: %C.de = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.b = fn_decl @Convert.7 [template = constants.%Convert.7c] {
// CHECK:STDOUT:     %self.patt: %C.71 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.71 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.71 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.6.%C [template = constants.%C.71]
// CHECK:STDOUT:     %self: %C.71 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.91 = fn_decl @Convert.8 [template = constants.%Convert.d4] {
// CHECK:STDOUT:     %self.patt: %C.a = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.a = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.a = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.7.%C [template = constants.%C.a]
// CHECK:STDOUT:     %self: %C.a = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.25 = fn_decl @Convert.9 [template = constants.%Convert.f02] {
// CHECK:STDOUT:     %self.patt: %C.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.1 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8.%C [template = constants.%C.1]
// CHECK:STDOUT:     %self: %C.1 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9e = fn_decl @Convert.10 [template = constants.%Convert.f0d] {
// CHECK:STDOUT:     %self.patt: %C.d0 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.d0 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: %C.d0 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.9.%C [template = constants.%C.d0]
// CHECK:STDOUT:     %self: %C.d0 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: %i32.builtin) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.8
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc7_16: %D.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc7_28: %D.elem = field_decl m, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m.5 [template = constants.%complete_type.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc8_31: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:   %int_0.loc8_39: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:   %.loc8_40.1: %struct_type.n.m.8 = struct_literal (%int_0.loc8_31, %int_0.loc8_39)
// CHECK:STDOUT:   %impl.elem0.loc8_40.1: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc8_40.1: <bound method> = bound_method %int_0.loc8_31, %impl.elem0.loc8_40.1 [template = constants.%Convert.bound.59]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.1: init %i32.builtin = call %Convert.bound.loc8_40.1(%int_0.loc8_31) [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_40.2: init %i32.builtin = converted %int_0.loc8_31, %int.convert_checked.loc8_40.1 [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_40.3: ref %i32.builtin = class_element_access %return, element0
// CHECK:STDOUT:   %.loc8_40.4: init %i32.builtin = initialize_from %.loc8_40.2 to %.loc8_40.3 [template = constants.%int_0.a]
// CHECK:STDOUT:   %impl.elem0.loc8_40.2: %Convert.type.2a = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc8_40.2: <bound method> = bound_method %int_0.loc8_39, %impl.elem0.loc8_40.2 [template = constants.%Convert.bound.59]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.2: init %i32.builtin = call %Convert.bound.loc8_40.2(%int_0.loc8_39) [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_40.5: init %i32.builtin = converted %int_0.loc8_39, %int.convert_checked.loc8_40.2 [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_40.6: ref %i32.builtin = class_element_access %return, element1
// CHECK:STDOUT:   %.loc8_40.7: init %i32.builtin = initialize_from %.loc8_40.5 to %.loc8_40.6 [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_40.8: init %D = class_init (%.loc8_40.4, %.loc8_40.7), %return [template = constants.%D.val]
// CHECK:STDOUT:   %.loc8_41: init %D = converted %.loc8_40.1, %.loc8_40.8 [template = constants.%D.val]
// CHECK:STDOUT:   return %.loc8_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.d) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.5]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.c]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.75]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.de]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.71]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.a]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.1]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc16: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc16
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.d0]() -> %return.param_patt: %D {
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
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2a
// CHECK:STDOUT:   %Convert => constants.%Convert.75
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.a
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_0.a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.8
// CHECK:STDOUT:   %Convert => constants.%Convert.b4
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.5d) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5d
// CHECK:STDOUT:   %Self.as_type => constants.%C.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_1.f
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_1.f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.4) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.4
// CHECK:STDOUT:   %Self.as_type => constants.%C.c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_2.5
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_2.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.0b) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.0b
// CHECK:STDOUT:   %Self.as_type => constants.%C.75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_3.a
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_3.a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.f) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.f
// CHECK:STDOUT:   %Self.as_type => constants.%C.de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_4.4
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_4.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.06) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.06
// CHECK:STDOUT:   %Self.as_type => constants.%C.71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.9) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_5.9
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_5.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.6) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.6
// CHECK:STDOUT:   %Self.as_type => constants.%C.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.e) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_6.e
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_6.e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.5b) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5b
// CHECK:STDOUT:   %Self.as_type => constants.%C.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_7.6
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_7.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.2) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.2
// CHECK:STDOUT:   %Self.as_type => constants.%C.d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [template]
// CHECK:STDOUT:   %complete_type.6: <witness> = complete_type_witness %struct_type.n.m [template]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [template]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.de: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.4: type = assoc_entity_type %ImplicitAs.type.d, %Convert.type.6 [symbolic]
// CHECK:STDOUT:   %assoc0.317cac.1: %Convert.assoc_type.4 = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.75: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.f: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.2 [template]
// CHECK:STDOUT:   %assoc0.6: %Convert.assoc_type.f = assoc_entity element0, imports.%import_ref.dbaf9d.1 [template]
// CHECK:STDOUT:   %assoc0.317cac.2: %Convert.assoc_type.4 = assoc_entity element0, imports.%import_ref.dbaf9d.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (imports.%import_ref.02) [template]
// CHECK:STDOUT:   %Convert.type.c8: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.71: %Convert.type.c8 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.59: <bound method> = bound_method %int_0.5, %Convert.71 [template]
// CHECK:STDOUT:   %int_0.a: %i32.builtin = int_value 0 [template]
// CHECK:STDOUT:   %C.1a: type = class_type @C, @C(%int_0.a) [template]
// CHECK:STDOUT:   %C.val.b: %C.1a = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.c1: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.2c: %Convert.type.c1 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.0: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.c1 [template]
// CHECK:STDOUT:   %assoc0.0: %Convert.assoc_type.0 = assoc_entity element0, imports.%import_ref.dbaf9d.1 [template]
// CHECK:STDOUT:   %int_1.f: %i32.builtin = int_value 1 [template]
// CHECK:STDOUT:   %C.b: type = class_type @C, @C(%int_1.f) [template]
// CHECK:STDOUT:   %int_2.5: %i32.builtin = int_value 2 [template]
// CHECK:STDOUT:   %C.1f: type = class_type @C, @C(%int_2.5) [template]
// CHECK:STDOUT:   %int_3.a: %i32.builtin = int_value 3 [template]
// CHECK:STDOUT:   %C.93: type = class_type @C, @C(%int_3.a) [template]
// CHECK:STDOUT:   %int_4.4: %i32.builtin = int_value 4 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%int_4.4) [template]
// CHECK:STDOUT:   %int_5.9: %i32.builtin = int_value 5 [template]
// CHECK:STDOUT:   %C.d: type = class_type @C, @C(%int_5.9) [template]
// CHECK:STDOUT:   %int_6.e: %i32.builtin = int_value 6 [template]
// CHECK:STDOUT:   %C.9d: type = class_type @C, @C(%int_6.e) [template]
// CHECK:STDOUT:   %int_7.6: %i32.builtin = int_value 7 [template]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%int_7.6) [template]
// CHECK:STDOUT:   %impl_witness.d: <witness> = impl_witness (imports.%import_ref.5) [template]
// CHECK:STDOUT:   %Convert.type.a9: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.a: %Convert.type.a9 = struct_value () [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.5f: <bound method> = bound_method %int_1.5, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.73: %C.b = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.9b: <witness> = impl_witness (imports.%import_ref.93) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.f: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %int_2.e: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.a: <bound method> = bound_method %int_2.e, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.3b: %C.1f = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.c: <witness> = impl_witness (imports.%import_ref.4a0) [template]
// CHECK:STDOUT:   %Convert.type.a2: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.21: %Convert.type.a2 = struct_value () [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.22: <bound method> = bound_method %int_3.1, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.8: %C.93 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.b: <witness> = impl_witness (imports.%import_ref.d4) [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.b: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %int_4.0: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.50: <bound method> = bound_method %int_4.0, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.7a: %C.3 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.9c: <witness> = impl_witness (imports.%import_ref.d2) [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.7a: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %int_5.6: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.2c: <bound method> = bound_method %int_5.6, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.313: %C.d = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.6: <witness> = impl_witness (imports.%import_ref.bd) [template]
// CHECK:STDOUT:   %Convert.type.c0: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.d0: %Convert.type.c0 = struct_value () [template]
// CHECK:STDOUT:   %int_6.4: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.d: <bound method> = bound_method %int_6.4, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.311: %C.9d = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.8: <witness> = impl_witness (imports.%import_ref.c0) [template]
// CHECK:STDOUT:   %Convert.type.f: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.f = struct_value () [template]
// CHECK:STDOUT:   %int_7.2: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.e: <bound method> = bound_method %int_7.2, %Convert.71 [template]
// CHECK:STDOUT:   %C.val.c: %C.7 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.a: <witness> = impl_witness (imports.%import_ref.05) [template]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.10 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.27
// CHECK:STDOUT:     .C = %import_ref.867
// CHECK:STDOUT:     .Make = %import_ref.30
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.71
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7f1
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.27: type = import_ref P//library, D, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.7e: <witness> = import_ref P//library, loc7_35, loaded [template = constants.%complete_type.6]
// CHECK:STDOUT:   %import_ref.ca = import_ref P//library, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.a9 = import_ref P//library, loc7_16, unloaded
// CHECK:STDOUT:   %import_ref.9d = import_ref P//library, loc7_28, unloaded
// CHECK:STDOUT:   %import_ref.867: %C.type = import_ref P//library, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref P//library, loc6_19, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.ad = import_ref P//library, inst42 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.01 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.86d: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.317cac.2)]
// CHECK:STDOUT:   %import_ref.03 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.dbaf9d.1 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %import_ref.4ae: <witness> = import_ref Core//default, loc12_38, loaded [template = constants.%impl_witness.2]
// CHECK:STDOUT:   %import_ref.af: type = import_ref Core//default, loc12_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.d93: type = import_ref Core//default, loc12_36, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.8fe: <witness> = import_ref P//library, loc10_33, loaded [template = constants.%impl_witness.d]
// CHECK:STDOUT:   %import_ref.d9f: type = import_ref P//library, loc10_9, loaded [template = constants.%C.1a]
// CHECK:STDOUT:   %import_ref.262642.1: type = import_ref P//library, loc10_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.1: <witness> = import_ref P//library, loc11_33, loaded [template = constants.%impl_witness.9b]
// CHECK:STDOUT:   %import_ref.7b: type = import_ref P//library, loc11_9, loaded [template = constants.%C.b]
// CHECK:STDOUT:   %import_ref.262642.2: type = import_ref P//library, loc11_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.8e: <witness> = import_ref P//library, loc12_33, loaded [template = constants.%impl_witness.c]
// CHECK:STDOUT:   %import_ref.bc: type = import_ref P//library, loc12_9, loaded [template = constants.%C.1f]
// CHECK:STDOUT:   %import_ref.262642.3: type = import_ref P//library, loc12_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.31: <witness> = import_ref P//library, loc13_33, loaded [template = constants.%impl_witness.b]
// CHECK:STDOUT:   %import_ref.6: type = import_ref P//library, loc13_9, loaded [template = constants.%C.93]
// CHECK:STDOUT:   %import_ref.262642.4: type = import_ref P//library, loc13_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.4f: <witness> = import_ref P//library, loc14_33, loaded [template = constants.%impl_witness.9c]
// CHECK:STDOUT:   %import_ref.22: type = import_ref P//library, loc14_9, loaded [template = constants.%C.3]
// CHECK:STDOUT:   %import_ref.262642.5: type = import_ref P//library, loc14_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.0f: <witness> = import_ref P//library, loc15_33, loaded [template = constants.%impl_witness.6]
// CHECK:STDOUT:   %import_ref.7fd: type = import_ref P//library, loc15_9, loaded [template = constants.%C.d]
// CHECK:STDOUT:   %import_ref.262642.6: type = import_ref P//library, loc15_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.04: <witness> = import_ref P//library, loc16_33, loaded [template = constants.%impl_witness.8]
// CHECK:STDOUT:   %import_ref.83: type = import_ref P//library, loc16_9, loaded [template = constants.%C.9d]
// CHECK:STDOUT:   %import_ref.262642.7: type = import_ref P//library, loc16_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.f: <witness> = import_ref P//library, loc17_33, loaded [template = constants.%impl_witness.a]
// CHECK:STDOUT:   %import_ref.79: type = import_ref P//library, loc17_9, loaded [template = constants.%C.7]
// CHECK:STDOUT:   %import_ref.262642.8: type = import_ref P//library, loc17_31, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT:   %import_ref.30: %Make.type = import_ref P//library, Make, loaded [template = constants.%Make]
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
// CHECK:STDOUT:     %n.patt: %i32.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32.builtin = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.27 [template = constants.%D]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.3 [template = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.3: type = converted %int.make_type_signed, %.loc7_10.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.de)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d), @ImplicitAs.%Convert.type (%Convert.type.6) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.4)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4) = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic = %assoc0 (constants.%assoc0.317cac.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.01
// CHECK:STDOUT:     .Convert = imports.%import_ref.86d
// CHECK:STDOUT:     witness = (imports.%import_ref.03)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.af as imports.%import_ref.d93 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.d9f as imports.%import_ref.262642.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.8fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.7b as imports.%import_ref.262642.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.bc as imports.%import_ref.262642.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.8e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.6 as imports.%import_ref.262642.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: imports.%import_ref.22 as imports.%import_ref.262642.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.7fd as imports.%import_ref.262642.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.83 as imports.%import_ref.262642.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: imports.%import_ref.79 as imports.%import_ref.262642.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.ca
// CHECK:STDOUT:   .n = imports.%import_ref.a9
// CHECK:STDOUT:   .m = imports.%import_ref.9d
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%N: %i32.builtin) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ad
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: %i32.builtin) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc8: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:   %impl.elem0.loc8_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc8_34: <bound method> = bound_method %int_0, %impl.elem0.loc8_34 [template = constants.%Convert.bound.59]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32.builtin = call %Convert.bound.loc8_34(%int_0) [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_0.a]
// CHECK:STDOUT:   %.loc8_34.2: %i32.builtin = converted %int_0, %.loc8_34.1 [template = constants.%int_0.a]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_0.a) [template = constants.%C.1a]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.1a = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.1a = class_init (), %.loc8_24.2 [template = constants.%C.val.b]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.1a = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.1a = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.d, element0 [template = constants.%Convert.a]
// CHECK:STDOUT:   %Convert.bound.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.1a = bind_value %.loc8_26.1
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
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %impl.elem0.loc9_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc9_34: <bound method> = bound_method %int_1, %impl.elem0.loc9_34 [template = constants.%Convert.bound.5f]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32.builtin = call %Convert.bound.loc9_34(%int_1) [template = constants.%int_1.f]
// CHECK:STDOUT:   %.loc9_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc9 [template = constants.%int_1.f]
// CHECK:STDOUT:   %.loc9_34.2: %i32.builtin = converted %int_1, %.loc9_34.1 [template = constants.%int_1.f]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_1.f) [template = constants.%C.b]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.b = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.b = class_init (), %.loc9_24.2 [template = constants.%C.val.73]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.b = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.b = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.9b, element0 [template = constants.%Convert.f]
// CHECK:STDOUT:   %Convert.bound.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.b = bind_value %.loc9_26.1
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
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.e]
// CHECK:STDOUT:   %impl.elem0.loc10_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc10_34: <bound method> = bound_method %int_2, %impl.elem0.loc10_34 [template = constants.%Convert.bound.a]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32.builtin = call %Convert.bound.loc10_34(%int_2) [template = constants.%int_2.5]
// CHECK:STDOUT:   %.loc10_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_2.5]
// CHECK:STDOUT:   %.loc10_34.2: %i32.builtin = converted %int_2, %.loc10_34.1 [template = constants.%int_2.5]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_2.5) [template = constants.%C.1f]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.1f = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.1f = class_init (), %.loc10_24.2 [template = constants.%C.val.3b]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.1f = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.1f = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.c, element0 [template = constants.%Convert.21]
// CHECK:STDOUT:   %Convert.bound.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.1f = bind_value %.loc10_26.1
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
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc11_34: <bound method> = bound_method %int_3, %impl.elem0.loc11_34 [template = constants.%Convert.bound.22]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32.builtin = call %Convert.bound.loc11_34(%int_3) [template = constants.%int_3.a]
// CHECK:STDOUT:   %.loc11_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_3.a]
// CHECK:STDOUT:   %.loc11_34.2: %i32.builtin = converted %int_3, %.loc11_34.1 [template = constants.%int_3.a]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_3.a) [template = constants.%C.93]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.93 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.93 = class_init (), %.loc11_24.2 [template = constants.%C.val.8]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.93 = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.93 = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.b, element0 [template = constants.%Convert.b]
// CHECK:STDOUT:   %Convert.bound.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.93 = bind_value %.loc11_26.1
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
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc12_34: <bound method> = bound_method %int_4, %impl.elem0.loc12_34 [template = constants.%Convert.bound.50]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32.builtin = call %Convert.bound.loc12_34(%int_4) [template = constants.%int_4.4]
// CHECK:STDOUT:   %.loc12_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_4.4]
// CHECK:STDOUT:   %.loc12_34.2: %i32.builtin = converted %int_4, %.loc12_34.1 [template = constants.%int_4.4]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_4.4) [template = constants.%C.3]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.3 = class_init (), %.loc12_24.2 [template = constants.%C.val.7a]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.3 = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.3 = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.9c, element0 [template = constants.%Convert.7a]
// CHECK:STDOUT:   %Convert.bound.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.3 = bind_value %.loc12_26.1
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
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.6]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc13_34: <bound method> = bound_method %int_5, %impl.elem0.loc13_34 [template = constants.%Convert.bound.2c]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32.builtin = call %Convert.bound.loc13_34(%int_5) [template = constants.%int_5.9]
// CHECK:STDOUT:   %.loc13_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_5.9]
// CHECK:STDOUT:   %.loc13_34.2: %i32.builtin = converted %int_5, %.loc13_34.1 [template = constants.%int_5.9]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_5.9) [template = constants.%C.d]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.d = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.d = class_init (), %.loc13_24.2 [template = constants.%C.val.313]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.d = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.d = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.6, element0 [template = constants.%Convert.d0]
// CHECK:STDOUT:   %Convert.bound.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.d = bind_value %.loc13_26.1
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
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.4]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc14_34: <bound method> = bound_method %int_6, %impl.elem0.loc14_34 [template = constants.%Convert.bound.d]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32.builtin = call %Convert.bound.loc14_34(%int_6) [template = constants.%int_6.e]
// CHECK:STDOUT:   %.loc14_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_6.e]
// CHECK:STDOUT:   %.loc14_34.2: %i32.builtin = converted %int_6, %.loc14_34.1 [template = constants.%int_6.e]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_6.e) [template = constants.%C.9d]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.9d = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.9d = class_init (), %.loc14_24.2 [template = constants.%C.val.311]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.9d = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.9d = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.8, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.9d = bind_value %.loc14_26.1
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
// CHECK:STDOUT:   %C.ref.loc15: %C.type = name_ref C, imports.%import_ref.867 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.2]
// CHECK:STDOUT:   %impl.elem0.loc15_34: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.71]
// CHECK:STDOUT:   %Convert.bound.loc15_34: <bound method> = bound_method %int_7, %impl.elem0.loc15_34 [template = constants.%Convert.bound.e]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32.builtin = call %Convert.bound.loc15_34(%int_7) [template = constants.%int_7.6]
// CHECK:STDOUT:   %.loc15_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_7.6]
// CHECK:STDOUT:   %.loc15_34.2: %i32.builtin = converted %int_7, %.loc15_34.1 [template = constants.%int_7.6]
// CHECK:STDOUT:   %C.loc15: type = class_type @C, @C(constants.%int_7.6) [template = constants.%C.7]
// CHECK:STDOUT:   %.loc15_24.2: ref %C.7 = temporary_storage
// CHECK:STDOUT:   %.loc15_24.3: init %C.7 = class_init (), %.loc15_24.2 [template = constants.%C.val.c]
// CHECK:STDOUT:   %.loc15_24.4: ref %C.7 = temporary %.loc15_24.2, %.loc15_24.3
// CHECK:STDOUT:   %.loc15_26.1: ref %C.7 = converted %.loc15_24.1, %.loc15_24.4
// CHECK:STDOUT:   %impl.elem0.loc15_35: %Convert.type.c1 = impl_witness_access constants.%impl_witness.a, element0 [template = constants.%Convert.5]
// CHECK:STDOUT:   %Convert.bound.loc15_35: <bound method> = bound_method %.loc15_26.1, %impl.elem0.loc15_35
// CHECK:STDOUT:   %.loc15_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc15_26.2: %C.7 = bind_value %.loc15_26.1
// CHECK:STDOUT:   %Convert.call.loc15: init %D = call %Convert.bound.loc15_35(%.loc15_26.2) to %.loc15_35.1
// CHECK:STDOUT:   %.loc15_35.2: init %D = converted %.loc15_26.1, %Convert.call.loc15
// CHECK:STDOUT:   return %.loc15_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %P.ref.loc16: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%import_ref.30 [template = constants.%Make]
// CHECK:STDOUT:   %.loc7_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.d) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %C.1a]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %C.b]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %C.1f]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: %C.93]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: %C.3]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: %C.d]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: %C.9d]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10[%self.param_patt: %C.7]() -> %D [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %D [from "library.carbon"];
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
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.75
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a) {
// CHECK:STDOUT:   %N => constants.%int_0.a
// CHECK:STDOUT:   %N.patt => constants.%int_0.a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c1
// CHECK:STDOUT:   %Convert => constants.%Convert.2c
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f) {
// CHECK:STDOUT:   %N => constants.%int_1.f
// CHECK:STDOUT:   %N.patt => constants.%int_1.f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5) {
// CHECK:STDOUT:   %N => constants.%int_2.5
// CHECK:STDOUT:   %N.patt => constants.%int_2.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a) {
// CHECK:STDOUT:   %N => constants.%int_3.a
// CHECK:STDOUT:   %N.patt => constants.%int_3.a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4) {
// CHECK:STDOUT:   %N => constants.%int_4.4
// CHECK:STDOUT:   %N.patt => constants.%int_4.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.9) {
// CHECK:STDOUT:   %N => constants.%int_5.9
// CHECK:STDOUT:   %N.patt => constants.%int_5.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.e) {
// CHECK:STDOUT:   %N => constants.%int_6.e
// CHECK:STDOUT:   %N.patt => constants.%int_6.e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6) {
// CHECK:STDOUT:   %N => constants.%int_7.6
// CHECK:STDOUT:   %N.patt => constants.%int_7.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
