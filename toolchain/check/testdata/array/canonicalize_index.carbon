// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/canonicalize_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/canonicalize_index.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn ConvertToU32(a: i32) -> u32 = "int.convert_checked";

var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;
let c: [i32; ConvertToU32(3)]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [template]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %.9: %.5 = assoc_entity element0, imports.%import_ref.8 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.10: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.10)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.12: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.12)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.21 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.34 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.20: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.21: %.20 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.10)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.12)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.24: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.25: %.24 = assoc_entity element0, imports.%import_ref.45 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.33, %Convert.16 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.37: type = array_type %.36, i32 [template]
// CHECK:STDOUT:   %.38: type = ptr_type %.37 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.39: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.36, %Convert.15 [template]
// CHECK:STDOUT:   %array: %.37 = tuple_value (%.30, %.32, %.33) [template]
// CHECK:STDOUT:   %.41: %.2 = int_value 3 [template]
// CHECK:STDOUT:   %.42: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.5, @impl.6(%.1) [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %.43: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %.44: <bound method> = bound_method %.41, %Convert.17 [template]
// CHECK:STDOUT:   %.45: <specific function> = specific_function %.44, @Convert.5(%.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Core//prelude/types, inst+45, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %import_ref.3: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%.1 (%.5) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.10: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.11: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.28]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.20: <witness> = import_ref Core//prelude/operators/as, inst+296, loaded [template = constants.%.34]
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.27: <witness> = import_ref Core//prelude/operators/as, inst+385, loaded [symbolic = @impl.6.%.2 (constants.%.17)]
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.30 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.32: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.38 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.40: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.41 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.43: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.45 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.47: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.48 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.50: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.51 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ConvertToU32 = %ConvertToU32.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertToU32.decl: %ConvertToU32.type = fn_decl @ConvertToU32 [template = constants.%ConvertToU32] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc12_20.2: type = converted %int.make_type_32, %.loc12_20.1 [template = i32]
// CHECK:STDOUT:     %.loc12_28.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc12_28.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_28.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_28.3: type = converted %int.make_type_unsigned, %.loc12_28.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc14_18.1: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_21.1: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_18.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_18.3: <bound method> = bound_method %.loc14_18.1, %.loc14_18.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc14_18: init i32 = call %.loc14_18.3(%.loc14_18.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_18.4: i32 = value_of_initializer %int.convert_checked.loc14_18 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_18.5: i32 = converted %.loc14_18.1, %.loc14_18.4 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_21.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_21.3: <bound method> = bound_method %.loc14_21.1, %.loc14_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc14_21: init i32 = call %.loc14_21.3(%.loc14_21.1) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_21.4: i32 = value_of_initializer %int.convert_checked.loc14_21 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_21.5: i32 = converted %.loc14_21.1, %.loc14_21.4 [template = constants.%.32]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc14_18.5, %.loc14_21.5) [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_9.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_9.2: type = converted %int.make_type_32.loc14, %.loc14_9.1 [template = i32]
// CHECK:STDOUT:   %.loc14_17.1: %Convert.type.5 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc14_17.2: <bound method> = bound_method %int.sadd, %.loc14_17.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_17.3: i32 = value_of_initializer %int.sadd [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_17.4: i32 = converted %int.sadd, %.loc14_17.3 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_17: init Core.IntLiteral = call %.loc14_17.2(%.loc14_17.4) [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_17.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_17 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_17.6: Core.IntLiteral = converted %int.sadd, %.loc14_17.5 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_23: type = array_type %.loc14_17.6, i32 [template = constants.%.37]
// CHECK:STDOUT:   %a.var: ref %.37 = var a
// CHECK:STDOUT:   %a: ref %.37 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_14: Core.IntLiteral = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc15_9.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_9.2: type = converted %int.make_type_32.loc15, %.loc15_9.1 [template = i32]
// CHECK:STDOUT:   %.loc15_15: type = array_type %.loc15_14, i32 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_16: type = ptr_type %.37 [template = constants.%.38]
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %ConvertToU32.ref: %ConvertToU32.type = name_ref ConvertToU32, %ConvertToU32.decl [template = constants.%ConvertToU32]
// CHECK:STDOUT:   %.loc16_27.1: Core.IntLiteral = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_27.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_27.3: <bound method> = bound_method %.loc16_27.1, %.loc16_27.2 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc16_27: init i32 = call %.loc16_27.3(%.loc16_27.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_27.4: i32 = value_of_initializer %int.convert_checked.loc16_27 [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_27.5: i32 = converted %.loc16_27.1, %.loc16_27.4 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc16_26.1: init %.2 = call %ConvertToU32.ref(%.loc16_27.5) [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_9.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_9.2: type = converted %int.make_type_32.loc16, %.loc16_9.1 [template = i32]
// CHECK:STDOUT:   %.loc16_26.1: %Convert.type.5 = interface_witness_access constants.%.43, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc16_26.2: <bound method> = bound_method %int.convert_checked.loc16_26.1, %.loc16_26.1 [template = constants.%.44]
// CHECK:STDOUT:   %.loc16_26.3: <specific function> = specific_function %.loc16_26.2, @Convert.5(constants.%.1) [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_26.4: %.2 = value_of_initializer %int.convert_checked.loc16_26.1 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_26.5: %.2 = converted %int.convert_checked.loc16_26.1, %.loc16_26.4 [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc16_26.2: init Core.IntLiteral = call %.loc16_26.3(%.loc16_26.5) [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_26.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc16_26.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_26.7: Core.IntLiteral = converted %int.convert_checked.loc16_26.1, %.loc16_26.6 [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_29: type = array_type %.loc16_26.7, i32 [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_30: type = ptr_type %.37 [template = constants.%.38]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.5) = assoc_entity element0, imports.%import_ref.7 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     .Convert = imports.%import_ref.5
// CHECK:STDOUT:     witness = (imports.%import_ref.6)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.18)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.18) = assoc_entity element0, imports.%import_ref.34 [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.28
// CHECK:STDOUT:     .Convert = imports.%import_ref.29
// CHECK:STDOUT:     witness = (imports.%import_ref.30)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.18 as imports.%import_ref.19 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.31 as imports.%import_ref.32 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.36 as imports.%import_ref.37 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.39 as imports.%import_ref.40 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.41
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.42 as imports.%import_ref.43 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.46 as imports.%import_ref.47 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.48
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.49 as imports.%import_ref.50 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.51
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%size.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToU32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.10) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.10)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.4)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.10) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.10)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.12[%self.param_patt: i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_28: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_31: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_34: Core.IntLiteral = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_35.1: %tuple.type = tuple_literal (%.loc14_28, %.loc14_31, %.loc14_34)
// CHECK:STDOUT:   %.loc14_35.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_35.3: <bound method> = bound_method %.loc14_28, %.loc14_35.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.1: init i32 = call %.loc14_35.3(%.loc14_28) [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_35.4: init i32 = converted %.loc14_28, %int.convert_checked.loc14_35.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_35.5: i32 = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %.loc14_35.6: ref i32 = array_index file.%a.var, %.loc14_35.5
// CHECK:STDOUT:   %.loc14_35.7: init i32 = initialize_from %.loc14_35.4 to %.loc14_35.6 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_35.8: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_35.9: <bound method> = bound_method %.loc14_31, %.loc14_35.8 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.2: init i32 = call %.loc14_35.9(%.loc14_31) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_35.10: init i32 = converted %.loc14_31, %int.convert_checked.loc14_35.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_35.11: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_35.12: ref i32 = array_index file.%a.var, %.loc14_35.11
// CHECK:STDOUT:   %.loc14_35.13: init i32 = initialize_from %.loc14_35.10 to %.loc14_35.12 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_35.14: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_35.15: <bound method> = bound_method %.loc14_34, %.loc14_35.14 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.3: init i32 = call %.loc14_35.15(%.loc14_34) [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_35.16: init i32 = converted %.loc14_34, %int.convert_checked.loc14_35.3 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_35.17: i32 = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_35.18: ref i32 = array_index file.%a.var, %.loc14_35.17
// CHECK:STDOUT:   %.loc14_35.19: init i32 = initialize_from %.loc14_35.16 to %.loc14_35.18 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_35.20: init %.37 = array_init (%.loc14_35.7, %.loc14_35.13, %.loc14_35.19) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_36: init %.37 = converted %.loc14_35.1, %.loc14_35.20 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc14_36
// CHECK:STDOUT:   %a.ref.loc15: ref %.37 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc15: %.38 = addr_of %a.ref.loc15
// CHECK:STDOUT:   %b: %.38 = bind_name b, %.loc15
// CHECK:STDOUT:   %a.ref.loc16: ref %.37 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc16: %.38 = addr_of %a.ref.loc16
// CHECK:STDOUT:   %c: %.38 = bind_name c, %.loc16
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.10) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
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
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.9
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %.2 => constants.%.21
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
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.10) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT:   %.2 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%.1) {
// CHECK:STDOUT:   %N => constants.%.1
// CHECK:STDOUT:   %N.patt => constants.%.1
// CHECK:STDOUT:   %.1 => constants.%.42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%.1) {
// CHECK:STDOUT:   %N => constants.%.1
// CHECK:STDOUT:   %N.patt => constants.%.1
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.17
// CHECK:STDOUT:   %Convert => constants.%Convert.17
// CHECK:STDOUT:   %.2 => constants.%.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%.1) {
// CHECK:STDOUT:   %N => constants.%.1
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
