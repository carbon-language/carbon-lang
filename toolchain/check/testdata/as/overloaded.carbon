// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/overloaded.carbon

class X {
  var n: i32;
}

impl i32 as Core.As(X) {
  fn Convert[self: i32]() -> X { return {.n = self}; }
}

impl X as Core.As(i32) {
  fn Convert[self: X]() -> i32 { return self.n; }
}

let n: i32 = ((4 as i32) as X) as i32;

// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %X, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %As.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%X)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @As(%X) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %As.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @As(i32) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %As.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.12: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.14: %.4 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Self.4: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.6 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.4, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.7 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.19: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.19)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %.21: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.21)> [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.4, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.23: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.10 [template]
// CHECK:STDOUT:   %.24: %.23 = assoc_entity element0, imports.%import_ref.25 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.7, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.8, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.19)> [symbolic]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.10(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(%.21)> [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %As.type.7: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.1, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: type = assoc_entity_type %As.type.7, %Convert.type.15 [template]
// CHECK:STDOUT:   %.30: %.29 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.11, @impl.13(%N) [symbolic]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [symbolic]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.16) [symbolic]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.12, @impl.14(%N) [symbolic]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [symbolic]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.17) [symbolic]
// CHECK:STDOUT:   %Convert.type.18: type = fn_type @Convert.13 [template]
// CHECK:STDOUT:   %Convert.18: %Convert.type.18 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.18) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.13, %Convert.18 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.35, %Convert.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %As.type.1 = import_ref Core//prelude/operators/as, inst+14, loaded [template = constants.%As]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.1 (%.4) = import_ref Core//prelude/operators/as, inst+43, loaded [symbolic = @As.%.2 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.5: @As.%Convert.type (%Convert.type.1) = import_ref Core//prelude/operators/as, inst+36, loaded [symbolic = @As.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+83, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+133, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.4.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.18 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.5.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.25 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.7.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.27: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.8.%.1 (constants.%.21)]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.31 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.4]
// CHECK:STDOUT:   %import_ref.34: <witness> = import_ref Core//prelude/operators/as, inst+420, loaded [template = constants.%.33]
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.10.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.11.%As.type (constants.%As.type.6)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.7]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.13.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.7]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.14.%.1 (constants.%.21)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.7]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.1 = name_ref As, imports.%import_ref.2 [template = constants.%As]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%X)> [template = constants.%As.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.1 = name_ref As, imports.%import_ref.2 [template = constants.%As]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_18.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc19_18.2: type = converted %int.make_type_32, %.loc19_18.1 [template = i32]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(i32)> [template = constants.%As.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32, %.loc23_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.4) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.6) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.15) = assoc_entity element0, imports.%import_ref.14 [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .Convert = imports.%import_ref.9
// CHECK:STDOUT:     witness = (imports.%import_ref.10)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc16_20.2: type = converted %int.make_type_32, %.loc16_20.1 [template = i32]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %self.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_24: <witness> = interface_witness (%Convert.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc15_24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X.ref as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4 = fn_decl @Convert.3 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: %X = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %X = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc20_28.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc20_28.2: type = converted %int.make_type_32, %.loc20_28.1 [template = i32]
// CHECK:STDOUT:     %self.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_24: <witness> = interface_witness (%Convert.decl) [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc19_24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.26 as imports.%import_ref.27 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.10(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.10.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.13(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.11, @impl.13(%N) [symbolic = %Convert.type (constants.%Convert.type.16)]
// CHECK:STDOUT:   %Convert: @impl.13.%Convert.type (%Convert.type.16) = struct_value () [symbolic = %Convert (constants.%Convert.16)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.14(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.12, @impl.14(%N) [symbolic = %Convert.type (constants.%Convert.type.17)]
// CHECK:STDOUT:   %Convert: @impl.14.%Convert.type (%Convert.type.17) = struct_value () [symbolic = %Convert (constants.%Convert.17)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.32)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.1 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: i32]() -> %return: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc16_51.1: %.2 = struct_literal (%self.ref)
// CHECK:STDOUT:   %.loc16_51.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc16_51.3: init i32 = initialize_from %self.ref to %.loc16_51.2
// CHECK:STDOUT:   %.loc16_51.4: init %X = class_init (%.loc16_51.3), %return
// CHECK:STDOUT:   %.loc16_52: init %X = converted %.loc16_51.1, %.loc16_51.4
// CHECK:STDOUT:   return %.loc16_52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %X]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %.1 = name_ref n, @X.%.loc12_8 [template = @X.%.loc12_8]
// CHECK:STDOUT:   %.loc20_45.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc20_45.2: i32 = bind_value %.loc20_45.1
// CHECK:STDOUT:   return %.loc20_45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%Dest: type, constants.%Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%Self (%Self.4)]() -> @Convert.4.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.5.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.6.%.1 (%.21) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.7.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.8.%.1 (%.21)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.9.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.10.%.1 (%.21) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.11.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.12.%.1 (%.21)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.13[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_16: Core.IntLiteral = int_value 4 [template = constants.%.13]
// CHECK:STDOUT:   %int.make_type_32.loc23_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_21.1: type = value_of_initializer %int.make_type_32.loc23_21 [template = i32]
// CHECK:STDOUT:   %.loc23_21.2: type = converted %int.make_type_32.loc23_21, %.loc23_21.1 [template = i32]
// CHECK:STDOUT:   %.loc23_18.1: %Convert.type.5 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.18]
// CHECK:STDOUT:   %.loc23_18.2: <bound method> = bound_method %.loc23_16, %.loc23_18.1 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc23_18.2(%.loc23_16) [template = constants.%.35]
// CHECK:STDOUT:   %.loc23_18.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.35]
// CHECK:STDOUT:   %.loc23_18.4: i32 = converted %.loc23_16, %.loc23_18.3 [template = constants.%.35]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc23_26.1: %Convert.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%Convert.2]
// CHECK:STDOUT:   %.loc23_26.2: <bound method> = bound_method %.loc23_18.4, %.loc23_26.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc23_26.3: ref %X = temporary_storage
// CHECK:STDOUT:   %Convert.call.loc23_26: init %X = call %.loc23_26.2(%.loc23_18.4) to %.loc23_26.3
// CHECK:STDOUT:   %.loc23_26.4: init %X = converted %.loc23_18.4, %Convert.call.loc23_26
// CHECK:STDOUT:   %int.make_type_32.loc23_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_32.loc23_35 [template = i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_32.loc23_35, %.loc23_35.1 [template = i32]
// CHECK:STDOUT:   %.loc23_32.1: %Convert.type.5 = interface_witness_access constants.%.12, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:   %.loc23_32.2: <bound method> = bound_method %.loc23_26.4, %.loc23_32.1
// CHECK:STDOUT:   %.loc23_26.5: ref %X = temporary %.loc23_26.3, %.loc23_26.4
// CHECK:STDOUT:   %.loc23_26.6: %X = bind_value %.loc23_26.5
// CHECK:STDOUT:   %Convert.call.loc23_32: init i32 = call %.loc23_32.2(%.loc23_26.6)
// CHECK:STDOUT:   %.loc23_32.3: i32 = value_of_initializer %Convert.call.loc23_32
// CHECK:STDOUT:   %.loc23_32.4: i32 = converted %.loc23_26.4, %.loc23_32.3
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc23_32.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%X) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:   %Dest.patt => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%X, i32) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(i32, constants.%X) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7
// CHECK:STDOUT:   %Convert => constants.%Convert.7
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.4.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.4.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(@impl.4.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.21) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.5.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%N) {
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
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.10
// CHECK:STDOUT:   %Convert => constants.%Convert.10
// CHECK:STDOUT:   %.1 => constants.%.23
// CHECK:STDOUT:   %.2 => constants.%.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(@impl.7.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.10.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(@impl.10.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.21) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.11.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.21
// CHECK:STDOUT:   %Dest.patt => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.15
// CHECK:STDOUT:   %Convert => constants.%Convert.15
// CHECK:STDOUT:   %.1 => constants.%.29
// CHECK:STDOUT:   %.2 => constants.%.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(@impl.13.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.14(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.14(@impl.14.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
