// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/make_type_unsigned.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/make_type_unsigned.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn UInt(n: i32) -> type = "int.make_type_unsigned";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: UInt(64)) -> UInt(64) {
  return n;
}

fn G(n: UInt(13)) -> UInt(13) {
  return n;
}

fn Symbolic(N:! i32, x: UInt(N)) -> UInt(N) {
  return x;
}

// --- fail_zero_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_zero_size.carbon:[[@LINE+4]]:8: error: integer type width of 0 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: UInt(0);
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
var n: UInt(0);

// --- fail_negative_size.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: error: integer type width of -1 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
var n: UInt(Negate(1));

// --- fail_oversized.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var m: UInt(1000000000);
// CHECK:STDERR:        ^~~~~
var m: UInt(1000000000);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UInt = %UInt.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %UInt.decl: %UInt.type = fn_decl @UInt [template = constants.%UInt] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_12.2: type = converted %int.make_type_32, %.loc4_12.1 [template = i32]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.6: %.2 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.7: type = int_type signed, %N.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.7)> [symbolic]
// CHECK:STDOUT:   %N.patt.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.9: type = int_type unsigned, %N.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.7)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 64 [template]
// CHECK:STDOUT:   %.28: type = int_type unsigned, %.27 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 13 [template]
// CHECK:STDOUT:   %.32: type = int_type unsigned, %.31 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N.2: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.33: type = int_type unsigned, %N.2 [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Int32 = %import_ref.51
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.2) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.25]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.9)]
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
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT:   %import_ref.51: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Symbolic = %Symbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %n.patt: %.28 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.28 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.28 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.28 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc6_9: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_14.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_14.3: <bound method> = bound_method %.loc6_14.1, %.loc6_14.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc6_14: init i32 = call %.loc6_14.3(%.loc6_14.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_14.4: i32 = value_of_initializer %int.convert_checked.loc6_14 [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_14.5: i32 = converted %.loc6_14.1, %.loc6_14.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_13: init type = call %UInt.ref.loc6_9(%.loc6_14.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_13 [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_13, %.loc6_16.1 [template = constants.%.28]
// CHECK:STDOUT:     %UInt.ref.loc6_22: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_27.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_27.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_27.3: <bound method> = bound_method %.loc6_27.1, %.loc6_27.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc6_27: init i32 = call %.loc6_27.3(%.loc6_27.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_27.4: i32 = value_of_initializer %int.convert_checked.loc6_27 [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_27.5: i32 = converted %.loc6_27.1, %.loc6_27.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_26: init type = call %UInt.ref.loc6_22(%.loc6_27.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_26 [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_26, %.loc6_29.1 [template = constants.%.28]
// CHECK:STDOUT:     %n.param: %.28 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.28 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.28 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %.32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc10_9: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_14.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc10_14.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_14.3: <bound method> = bound_method %.loc10_14.1, %.loc10_14.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc10_14: init i32 = call %.loc10_14.3(%.loc10_14.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_14.4: i32 = value_of_initializer %int.convert_checked.loc10_14 [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_14.5: i32 = converted %.loc10_14.1, %.loc10_14.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_13: init type = call %UInt.ref.loc10_9(%.loc10_14.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_13 [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_13, %.loc10_16.1 [template = constants.%.32]
// CHECK:STDOUT:     %UInt.ref.loc10_22: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_27.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc10_27.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_27.3: <bound method> = bound_method %.loc10_27.1, %.loc10_27.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc10_27: init i32 = call %.loc10_27.3(%.loc10_27.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_27.4: i32 = value_of_initializer %int.convert_checked.loc10_27 [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_27.5: i32 = converted %.loc10_27.1, %.loc10_27.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_26: init type = call %UInt.ref.loc10_22(%.loc10_27.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_26 [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_26, %.loc10_29.1 [template = constants.%.32]
// CHECK:STDOUT:     %n.param: %.32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%.loc14_29 (%.33) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%.loc14_29 (%.33) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%.loc14_29 (%.33) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%.loc14_29 (%.33) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %int.make_type_32, %.loc14_17.1 [template = i32]
// CHECK:STDOUT:     %UInt.ref.loc14_25: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_30: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_29: init type = call %UInt.ref.loc14_25(%N.ref.loc14_30) [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_31.1: type = value_of_initializer %int.make_type_unsigned.loc14_29 [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_31.2: type = converted %int.make_type_unsigned.loc14_29, %.loc14_31.1 [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %UInt.ref.loc14_37: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_42: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_41: init type = call %UInt.ref.loc14_37(%N.ref.loc14_42) [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_43.1: type = value_of_initializer %int.make_type_unsigned.loc14_41 [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_43.2: type = converted %int.make_type_unsigned.loc14_41, %.loc14_43.1 [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %x.param: @Symbolic.%.loc14_29 (%.33) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%.loc14_29 (%.33) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%.loc14_29 (%.33) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%.loc14_29 (%.33) = return_slot %return.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.3)]
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
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.15) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
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
// CHECK:STDOUT: generic fn @Convert.7(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.1)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.28) -> %.28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.28 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.32) -> %.32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: i32) {
// CHECK:STDOUT:   %N.loc14_13.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %.loc14_29: type = int_type unsigned, %N.loc14_13.2 [symbolic = %.loc14_29 (constants.%.33)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.loc14_29 (%.33)) -> @Symbolic.%.loc14_29 (%.33) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%.loc14_29 (%.33) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
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
// CHECK:STDOUT: specific @As(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %N.patt => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N.1) {
// CHECK:STDOUT:   %N => constants.%N.1
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N.2) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %.loc14_29 => constants.%.33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.6: %.2 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.7: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.7)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.9: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.7)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.2) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.25]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.9)]
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
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:   %.loc10_13.1: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_13.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_13.3: <bound method> = bound_method %.loc10_13.1, %.loc10_13.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc10_13.3(%.loc10_13.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_13.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_13.5: i32 = converted %.loc10_13.1, %.loc10_13.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc10_13.5) [template = <error>]
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc10_14.2: type = converted %int.make_type_unsigned, %.loc10_14.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.3)]
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
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.15) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
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
// CHECK:STDOUT: specific @As(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %.6: %.2 = assoc_entity element0, imports.%import_ref.8 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.7: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.7)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.9: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.21 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.34 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.7)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.45 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.28: i32 = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.3: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%.1 (%.2) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.10: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.11: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.25]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+385, unloaded
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
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.47: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.48 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.50: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.51 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_32.loc6_14 [template = i32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_32.loc6_14, %.loc6_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int.make_type_32.loc6_22 [template = i32]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int.make_type_32.loc6_22, %.loc6_22.1 [template = i32]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_20.1: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_20.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_20.3: <bound method> = bound_method %.loc12_20.1, %.loc12_20.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc12_20.3(%.loc12_20.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_20.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_20.5: i32 = converted %.loc12_20.1, %.loc12_20.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_20.5) [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_19.1: i32 = value_of_initializer %int.snegate [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_19.2: i32 = converted %int.snegate, %.loc12_19.1 [template = constants.%.28]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc12_19.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_unsigned, %.loc12_22.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.7 [symbolic = %.2 (constants.%.3)]
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.15) = assoc_entity element0, imports.%import_ref.34 [symbolic = %.2 (constants.%.16)]
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.8)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.10)]
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.14)]
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.19)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
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
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.49 as imports.%import_ref.50 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.51
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
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
// CHECK:STDOUT: specific @As(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.6: %.2 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.7: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.7)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.9: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.7)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.9)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.2) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.25]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.9)]
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
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:   %.loc9_13.1: Core.IntLiteral = int_value 1000000000 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_13.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc9_13.3: <bound method> = bound_method %.loc9_13.1, %.loc9_13.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc9_13.3(%.loc9_13.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_13.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_13.5: i32 = converted %.loc9_13.1, %.loc9_13.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc9_13.5) [template = <error>]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.3)]
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
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.15) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
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
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.7) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.9) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.7)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.9)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
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
// CHECK:STDOUT: specific @As(constants.%.7) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.7
// CHECK:STDOUT:   %Dest.patt => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.9) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.9
// CHECK:STDOUT:   %Dest.patt => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
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
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
