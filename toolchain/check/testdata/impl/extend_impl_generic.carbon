// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon

// --- extend_impl_generic_interface.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F() -> T;
}

class Param {
  var x: i32;
}

class C {
  extend impl as HasF(Param) {
    fn F() -> Param {
      return {.x = 2};
    }
  }
}

fn G(c: C) {
  let a: i32 = C.F().x;
  var b: i32 = c.F().x;
}

// --- extend_impl_generic_class.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F[self: Self](t: T);
}

class X(U:! type) {
  extend impl as I(U) {
    fn F[self: Self](t: U) { }
  }
}

// CHECK:STDOUT: --- extend_impl_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.1: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %HasF: %HasF.type.1 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.2: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Param, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type.3: type = facet_type <@HasF, @HasF(%Param)> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @HasF(%Param) [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %HasF.type.3, %F.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %.9: type = struct_type {} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.12: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.13: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.3: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.17: %.16 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.18: %.14 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.19: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.19)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.21: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.21)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.23: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.24: %.23 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.4: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.5: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.28: %.27 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.29: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.30: %.29 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.19)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.21)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.10(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.33: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.34: %.33 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.13(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.36: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.12, %Convert.15 [template]
// CHECK:STDOUT:   %.39: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %Param = struct_value (%.39) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.40: type = ptr_type %.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.14) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.18)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.37]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.4.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.6.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.7.%.1 (constants.%.21)]
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.26 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.32 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.9.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.10.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.12.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.13.%.1 (constants.%.21)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.1 = interface_decl @HasF [template = constants.%HasF] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [template = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.2)]
// CHECK:STDOUT:   %Self.2: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc5_14.2: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.2), @HasF.%F.type (%F.type.1) [symbolic = %.loc5_14.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_14.3: @HasF.%.loc5_14.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_14.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.1)] {
// CHECK:STDOUT:       %return.patt: @F.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5_14.1: @HasF.%.loc5_14.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_14.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc5_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.14) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.27)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.27) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %return.patt: %Param = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Param = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.10(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.10.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.32)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.35)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.13(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.13(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.13.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc9_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc9_10.2: type = converted %int.make_type_32, %.loc9_10.1 [template = i32]
// CHECK:STDOUT:   %.loc9_8: %.3 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [template = constants.%HasF.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.1.%HasF.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_20: Core.IntLiteral = int_value 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc15_21.1: %.13 = struct_literal (%.loc15_20)
// CHECK:STDOUT:   %.loc15_21.2: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_21.3: <bound method> = bound_method %.loc15_20, %.loc15_21.2 [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc15_21.3(%.loc15_20) [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_21.4: init i32 = converted %.loc15_20, %int.convert_checked [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_21.5: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_21.6: init i32 = initialize_from %.loc15_21.4 to %.loc15_21.5 [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_21.7: init %Param = class_init (%.loc15_21.6), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_22: init %Param = converted %.loc15_21.1, %.loc15_21.7 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc15_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.3)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.21) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.21)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.4: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.5)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.21) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.21)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_32.loc21, %.loc21_10.1 [template = i32]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc21_17.1: %.6 = specific_constant @HasF.%.loc5_14.1, @HasF(constants.%Param) [template = constants.%.7]
// CHECK:STDOUT:   %F.ref.loc21: %.6 = name_ref F, %.loc21_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_17.2: %F.type.2 = interface_witness_access constants.%.8, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc21_19.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc21: init %Param = call %.loc21_17.2() to %.loc21_19.1
// CHECK:STDOUT:   %.loc21_19.2: ref %Param = temporary %.loc21_19.1, %F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %.3 = name_ref x, @Param.%.loc9_8 [template = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc21_21.1: ref i32 = class_element_access %.loc21_19.2, element0
// CHECK:STDOUT:   %.loc21_21.2: i32 = bind_value %.loc21_21.1
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc21_21.2
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_10.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_10.2: type = converted %int.make_type_32.loc22, %.loc22_10.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_17.1: %.6 = specific_constant @HasF.%.loc5_14.1, @HasF(constants.%Param) [template = constants.%.7]
// CHECK:STDOUT:   %F.ref.loc22: %.6 = name_ref F, %.loc22_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_17.2: %F.type.2 = interface_witness_access constants.%.8, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc22_19.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc22: init %Param = call %.loc22_17.2() to %.loc22_19.1
// CHECK:STDOUT:   %.loc22_19.2: ref %Param = temporary %.loc22_19.1, %F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %.3 = name_ref x, @Param.%.loc9_8 [template = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc22_21.1: ref i32 = class_element_access %.loc22_19.2, element0
// CHECK:STDOUT:   %.loc22_21.2: i32 = bind_value %.loc22_21.1
// CHECK:STDOUT:   assign %b.var, %.loc22_21.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@HasF.%T.loc4_16.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc5_14.2 => constants.%.6
// CHECK:STDOUT:   %.loc5_14.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Param, constants.%C) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.21) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.4.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(@impl.4.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.23
// CHECK:STDOUT:   %.2 => constants.%.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(@impl.7.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.9
// CHECK:STDOUT:   %.1 => constants.%.29
// CHECK:STDOUT:   %.2 => constants.%.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.4) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.21) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.10.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(@impl.10.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.33
// CHECK:STDOUT:   %.2 => constants.%.34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(@impl.13.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [template]
// CHECK:STDOUT:   %X.1: %X.type = struct_value () [template]
// CHECK:STDOUT:   %X.2: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %I.type.3, %F.type.2 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.3) [symbolic]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [template = constants.%X.1] {
// CHECK:STDOUT:     %U.patt.loc8_9.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc8_9.1, runtime_param<invalid> [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc8_9.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc5_25.2: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %.loc5_25.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_25.3: @I.%.loc5_25.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_25.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self (%Self) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %t.patt: @F.1.%T (%T) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.1.%T (%T) = value_param_pattern %t.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_14.1: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc5_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc5_14.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc5_14.3: type = converted %Self.ref, %.loc5_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.1.%Self (%Self) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.1.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.1.%T (%T) = value_param runtime_param1
// CHECK:STDOUT:       %t: @F.1.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5_25.1: @I.%.loc5_25.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_25.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc5_25.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X.2)]
// CHECK:STDOUT:   %I.type.loc9_19.2: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_19.2 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%U) [symbolic = %F.type (constants.%F.type.3)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.3) = struct_value () [symbolic = %F (constants.%F.3)]
// CHECK:STDOUT:   %.loc9_23.2: <witness> = interface_witness (%F) [symbolic = %.loc9_23.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_19.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.3) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.3)] {
// CHECK:STDOUT:       %self.patt: @F.2.%X (%X.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%X (%X.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %t.patt: @F.2.%U (%U) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.2.%U (%U) = value_param_pattern %t.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10: type = specific_constant constants.%X.2, @X(constants.%U) [symbolic = %X (constants.%X.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc10 [symbolic = %X (constants.%X.2)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.2.%X (%X.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%X (%X.2) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.2.%U (%U) = value_param runtime_param1
// CHECK:STDOUT:       %t: @F.2.%U (%U) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_23.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc9_23.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc9_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U.loc8_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_9.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.2)> [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl [template] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X.2 [symbolic = %X (constants.%X.2)]
// CHECK:STDOUT:       %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_19.1: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_19.2 (constants.%I.type.3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: type = specific_constant @impl.%I.type.loc9_19.1, @impl(constants.%U) [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:     %.loc12: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X.2
// CHECK:STDOUT:     extend %.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self (%Self)](%t.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%X (%X.2)](%t.param_patt: @F.2.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc5_25.2 => constants.%.3
// CHECK:STDOUT:   %.loc5_25.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@impl.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X.2
// CHECK:STDOUT:   %I.type.loc9_19.2 => constants.%I.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc9_23.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@F.2.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%X.2) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%X.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@impl.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X.2
// CHECK:STDOUT:   %I.type.loc9_19.2 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@X.%U.loc8_9.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
