// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/method.carbon

interface I {
  fn F[self: Self](other: Self) -> Self;
}

impl i32 as I {
  fn F[self: i32](other: i32) -> i32 = "int.sadd";
}

var arr: [i32; (1 as i32).(I.F)(2)];

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.3: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %As.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %As.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.9: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.4: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Self.5: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.3 [symbolic]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.4 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.14: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.14)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: <witness> = interface_witness (%Convert.5) [symbolic]
// CHECK:STDOUT:   %.16: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.16)> [symbolic]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.7 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.25 [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.21: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.14)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.16)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.10(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.1, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.24: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.25: %.24 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.13(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %F.2 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.33: %.10 = assoc_entity element0, imports.%import_ref.52 [symbolic]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.32, %Convert.16 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.13 [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %.38: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.37, %Convert.17 [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.41: type = array_type %.40, i32 [template]
// CHECK:STDOUT:   %.42: type = ptr_type %.41 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.51
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %As.type.1 = import_ref Core//prelude/operators/as, inst+14, loaded [template = constants.%As]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.1 (%.5) = import_ref Core//prelude/operators/as, inst+43, loaded [symbolic = @As.%.2 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.9: @ImplicitAs.%.1 (%.10) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.33)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.13: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.34]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.18 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.4.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.24: <witness> = import_ref Core//prelude/operators/as, inst+296, loaded [template = constants.%.38]
// CHECK:STDOUT:   %import_ref.25 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.6.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.27: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.7.%.1 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.31 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.34: <witness> = import_ref Core//prelude/operators/as, inst+420, loaded [template = constants.%.28]
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
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.12.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.13.%.1 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT:   %import_ref.51: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.52 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc19_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_17: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_32.loc19_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_22.1: type = value_of_initializer %int.make_type_32.loc19_22 [template = i32]
// CHECK:STDOUT:   %.loc19_22.2: type = converted %int.make_type_32.loc19_22, %.loc19_22.1 [template = i32]
// CHECK:STDOUT:   %.loc19_19.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc19_19.2: <bound method> = bound_method %.loc19_17, %.loc19_19.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc19_19: init i32 = call %.loc19_19.2(%.loc19_17) [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_19.3: i32 = value_of_initializer %int.convert_checked.loc19_19 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_19.4: i32 = converted %.loc19_17, %.loc19_19.3 [template = constants.%.30]
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @I.%.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_26.1: %F.type.1 = interface_witness_access constants.%.3, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %.loc19_26.2: <bound method> = bound_method %.loc19_19.4, %.loc19_26.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_33.1: Core.IntLiteral = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_33.2: %Convert.type.4 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc19_33.3: <bound method> = bound_method %.loc19_33.1, %.loc19_33.2 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc19_33: init i32 = call %.loc19_33.3(%.loc19_33.1) [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_33.4: i32 = value_of_initializer %int.convert_checked.loc19_33 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_33.5: i32 = converted %.loc19_33.1, %.loc19_33.4 [template = constants.%.36]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc19_26.2(%.loc19_19.4, %.loc19_33.5) [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_11.1: type = value_of_initializer %int.make_type_32.loc19_11 [template = i32]
// CHECK:STDOUT:   %.loc19_11.2: type = converted %int.make_type_32.loc19_11, %.loc19_11.1 [template = i32]
// CHECK:STDOUT:   %.loc19_32.1: %Convert.type.7 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc19_32.2: <bound method> = bound_method %int.sadd, %.loc19_32.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_32.3: i32 = value_of_initializer %int.sadd [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_32.4: i32 = converted %int.sadd, %.loc19_32.3 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc19_32: init Core.IntLiteral = call %.loc19_32.2(%.loc19_32.4) [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_32.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_32 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_32.6: Core.IntLiteral = converted %int.sadd, %.loc19_32.5 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_35: type = array_type %.loc19_32.6, i32 [template = constants.%.41]
// CHECK:STDOUT:   %arr.var: ref %.41 = var arr
// CHECK:STDOUT:   %arr: ref %.41 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self (%Self.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self (%Self.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @F.1.%Self (%Self.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @F.1.%Self (%Self.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self (%Self.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self (%Self.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc12_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_27.1: type = facet_type_access %Self.ref.loc12_27 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_27.2: type = converted %Self.ref.loc12_27, %.loc12_27.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc12_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_36.1: type = facet_type_access %Self.ref.loc12_36 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %Self.ref.loc12_36, %.loc12_36.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %self.param: @F.1.%Self (%Self.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%Self (%Self.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%Self (%Self.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @F.1.%Self (%Self.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self (%Self.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self (%Self.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.6)]
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
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.3) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.10) = assoc_entity element0, imports.%import_ref.14 [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .Convert = imports.%import_ref.9
// CHECK:STDOUT:     witness = (imports.%import_ref.10)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: i32 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: i32 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc16_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_14.1: type = value_of_initializer %int.make_type_32.loc16_14 [template = i32]
// CHECK:STDOUT:     %.loc16_14.2: type = converted %int.make_type_32.loc16_14, %.loc16_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc16_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_26.1: type = value_of_initializer %int.make_type_32.loc16_26 [template = i32]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %int.make_type_32.loc16_26, %.loc16_26.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc16_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_34.1: type = value_of_initializer %int.make_type_32.loc16_34 [template = i32]
// CHECK:STDOUT:     %.loc16_34.2: type = converted %int.make_type_32.loc16_34, %.loc16_34.1 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %other: i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15: <witness> = interface_witness (%F.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc15_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.5)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.5) = struct_value () [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.26 as imports.%import_ref.27 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.22)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.10(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.10.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.13(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.13.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self (%Self.1)](%other.param_patt: @F.1.%Self (%Self.1)) -> @F.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: i32](%other.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.2: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.3)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%Dest: type, constants.%Self.4: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self (%Self.5)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.4.%.1 (%.16) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%.1 (%.16)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.16) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.16)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.12[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.13[%self.param_patt: i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(i32) {
// CHECK:STDOUT:   %Self => i32
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
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
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
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %.2 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.4) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.16) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.4.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(@impl.4.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7
// CHECK:STDOUT:   %Convert => constants.%Convert.7
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(@impl.7.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.16) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.10.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(@impl.10.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT:   %.2 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(@impl.13.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
