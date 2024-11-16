// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/make_type_signed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/make_type_signed.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn Int(n: i32) -> type = "int.make_type_signed";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: Int(64)) -> Int(64) {
  return n;
}

fn G(n: Int(13)) -> Int(13) {
  return n;
}

fn Symbolic(N:! i32, x: Int(N)) -> Int(N) {
  return x;
}

// --- import_types.carbon

library "[[@TEST_NAME]]";

import library "types";
import library "use_types";

fn UseF(n: Int(64)) -> Int(64) {
  return F(n);
}

fn UseG(n: Int(13)) -> Int(13) {
  return G(n);
}

fn UseSymbolic(n: Int(24)) -> Int(24) {
  return Symbolic(24, n);
}

// --- fail_zero_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_zero_size.carbon:[[@LINE+4]]:8: error: integer type width of 0 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: Int(0);
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
var n: Int(0);

// --- fail_negative_size.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: error: integer type width of -1 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: Int(Negate(1));
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
var n: Int(Negate(1));

// --- fail_oversized.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var m: Int(1000000000);
// CHECK:STDERR:        ^~~~
var m: Int(1000000000);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 64 [template]
// CHECK:STDOUT:   %.28: type = int_type signed, %.27 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 13 [template]
// CHECK:STDOUT:   %.32: type = int_type signed, %.31 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N.2: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.33: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Int32 = %import_ref.51
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref.1
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
// CHECK:STDOUT:     %Int.ref.loc6_9: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc6_13.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_13.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_13.3: <bound method> = bound_method %.loc6_13.1, %.loc6_13.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc6_13: init i32 = call %.loc6_13.3(%.loc6_13.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_13.4: i32 = value_of_initializer %int.convert_checked.loc6_13 [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_13.5: i32 = converted %.loc6_13.1, %.loc6_13.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_signed.loc6_12: init type = call %Int.ref.loc6_9(%.loc6_13.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_15.1: type = value_of_initializer %int.make_type_signed.loc6_12 [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %int.make_type_signed.loc6_12, %.loc6_15.1 [template = constants.%.28]
// CHECK:STDOUT:     %Int.ref.loc6_21: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc6_25.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_25.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_25.3: <bound method> = bound_method %.loc6_25.1, %.loc6_25.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc6_25: init i32 = call %.loc6_25.3(%.loc6_25.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_25.4: i32 = value_of_initializer %int.convert_checked.loc6_25 [template = constants.%.27]
// CHECK:STDOUT:     %.loc6_25.5: i32 = converted %.loc6_25.1, %.loc6_25.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_signed.loc6_24: init type = call %Int.ref.loc6_21(%.loc6_25.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_27.1: type = value_of_initializer %int.make_type_signed.loc6_24 [template = constants.%.28]
// CHECK:STDOUT:     %.loc6_27.2: type = converted %int.make_type_signed.loc6_24, %.loc6_27.1 [template = constants.%.28]
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
// CHECK:STDOUT:     %Int.ref.loc10_9: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc10_13.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc10_13.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_13.3: <bound method> = bound_method %.loc10_13.1, %.loc10_13.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc10_13: init i32 = call %.loc10_13.3(%.loc10_13.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_13.4: i32 = value_of_initializer %int.convert_checked.loc10_13 [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_13.5: i32 = converted %.loc10_13.1, %.loc10_13.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_signed.loc10_12: init type = call %Int.ref.loc10_9(%.loc10_13.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_15.1: type = value_of_initializer %int.make_type_signed.loc10_12 [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_15.2: type = converted %int.make_type_signed.loc10_12, %.loc10_15.1 [template = constants.%.32]
// CHECK:STDOUT:     %Int.ref.loc10_21: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc10_25.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc10_25.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_25.3: <bound method> = bound_method %.loc10_25.1, %.loc10_25.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc10_25: init i32 = call %.loc10_25.3(%.loc10_25.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_25.4: i32 = value_of_initializer %int.convert_checked.loc10_25 [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_25.5: i32 = converted %.loc10_25.1, %.loc10_25.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_signed.loc10_24: init type = call %Int.ref.loc10_21(%.loc10_25.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_24 [template = constants.%.32]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_24, %.loc10_27.1 [template = constants.%.32]
// CHECK:STDOUT:     %n.param: %.32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%.loc14_28 (%.33) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%.loc14_28 (%.33) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%.loc14_28 (%.33) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%.loc14_28 (%.33) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %int.make_type_32, %.loc14_17.1 [template = i32]
// CHECK:STDOUT:     %Int.ref.loc14_25: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_29: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_28: init type = call %Int.ref.loc14_25(%N.ref.loc14_29) [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_30.1: type = value_of_initializer %int.make_type_signed.loc14_28 [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %int.make_type_signed.loc14_28, %.loc14_30.1 [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %Int.ref.loc14_36: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_40: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_39: init type = call %Int.ref.loc14_36(%N.ref.loc14_40) [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %int.make_type_signed.loc14_39 [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %int.make_type_signed.loc14_39, %.loc14_41.1 [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %x.param: @Symbolic.%.loc14_28 (%.33) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%.loc14_28 (%.33) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%.loc14_28 (%.33) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%.loc14_28 (%.33) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
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
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: i32) {
// CHECK:STDOUT:   %N.loc14_13.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %.loc14_28: type = int_type signed, %N.loc14_13.2 [symbolic = %.loc14_28 (constants.%.33)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.loc14_28 (%.33)) -> @Symbolic.%.loc14_28 (%.33) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%.loc14_28 (%.33) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N.2) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %.loc14_28 => constants.%.33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 64 [template]
// CHECK:STDOUT:   %.28: type = int_type signed, %.27 [template]
// CHECK:STDOUT:   %UseF.type: type = fn_type @UseF [template]
// CHECK:STDOUT:   %UseF: %UseF.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 13 [template]
// CHECK:STDOUT:   %.32: type = int_type signed, %.31 [template]
// CHECK:STDOUT:   %UseG.type: type = fn_type @UseG [template]
// CHECK:STDOUT:   %UseG: %UseG.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 24 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.33, %Convert.15 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 24 [template]
// CHECK:STDOUT:   %.36: type = int_type signed, %.35 [template]
// CHECK:STDOUT:   %UseSymbolic.type: type = fn_type @UseSymbolic [template]
// CHECK:STDOUT:   %UseSymbolic: %UseSymbolic.type = struct_value () [template]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT:   %N.2: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.37: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %N.patt.2: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %Symbolic, @Symbolic(%.35) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %import_ref.2: %F.type = import_ref Main//use_types, inst+405, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.3: %G.type = import_ref Main//use_types, inst+442, loaded [template = constants.%G]
// CHECK:STDOUT:   %import_ref.4: %Symbolic.type = import_ref Main//use_types, inst+482, loaded [template = constants.%Symbolic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref.1
// CHECK:STDOUT:     .F = imports.%import_ref.2
// CHECK:STDOUT:     .G = imports.%import_ref.3
// CHECK:STDOUT:     .Symbolic = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UseF = %UseF.decl
// CHECK:STDOUT:     .UseG = %UseG.decl
// CHECK:STDOUT:     .UseSymbolic = %UseSymbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UseF.decl: %UseF.type = fn_decl @UseF [template = constants.%UseF] {
// CHECK:STDOUT:     %n.patt: %.28 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.28 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.28 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.28 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc7_12: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc7_16.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc7_16.3: <bound method> = bound_method %.loc7_16.1, %.loc7_16.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc7_16: init i32 = call %.loc7_16.3(%.loc7_16.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc7_16.4: i32 = value_of_initializer %int.convert_checked.loc7_16 [template = constants.%.27]
// CHECK:STDOUT:     %.loc7_16.5: i32 = converted %.loc7_16.1, %.loc7_16.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_signed.loc7_15: init type = call %Int.ref.loc7_12(%.loc7_16.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc7_18.1: type = value_of_initializer %int.make_type_signed.loc7_15 [template = constants.%.28]
// CHECK:STDOUT:     %.loc7_18.2: type = converted %int.make_type_signed.loc7_15, %.loc7_18.1 [template = constants.%.28]
// CHECK:STDOUT:     %Int.ref.loc7_24: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc7_28.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_28.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc7_28.3: <bound method> = bound_method %.loc7_28.1, %.loc7_28.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc7_28: init i32 = call %.loc7_28.3(%.loc7_28.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc7_28.4: i32 = value_of_initializer %int.convert_checked.loc7_28 [template = constants.%.27]
// CHECK:STDOUT:     %.loc7_28.5: i32 = converted %.loc7_28.1, %.loc7_28.4 [template = constants.%.27]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call %Int.ref.loc7_24(%.loc7_28.5) [template = constants.%.28]
// CHECK:STDOUT:     %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%.28]
// CHECK:STDOUT:     %.loc7_30.2: type = converted %int.make_type_signed.loc7_27, %.loc7_30.1 [template = constants.%.28]
// CHECK:STDOUT:     %n.param: %.28 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.28 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.28 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseG.decl: %UseG.type = fn_decl @UseG [template = constants.%UseG] {
// CHECK:STDOUT:     %n.patt: %.32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc11_12: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc11_16.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc11_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc11_16.3: <bound method> = bound_method %.loc11_16.1, %.loc11_16.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc11_16: init i32 = call %.loc11_16.3(%.loc11_16.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc11_16.4: i32 = value_of_initializer %int.convert_checked.loc11_16 [template = constants.%.31]
// CHECK:STDOUT:     %.loc11_16.5: i32 = converted %.loc11_16.1, %.loc11_16.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_signed.loc11_15: init type = call %Int.ref.loc11_12(%.loc11_16.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%.32]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %int.make_type_signed.loc11_15, %.loc11_18.1 [template = constants.%.32]
// CHECK:STDOUT:     %Int.ref.loc11_24: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc11_28.1: Core.IntLiteral = int_value 13 [template = constants.%.29]
// CHECK:STDOUT:     %.loc11_28.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc11_28.3: <bound method> = bound_method %.loc11_28.1, %.loc11_28.2 [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc11_28: init i32 = call %.loc11_28.3(%.loc11_28.1) [template = constants.%.31]
// CHECK:STDOUT:     %.loc11_28.4: i32 = value_of_initializer %int.convert_checked.loc11_28 [template = constants.%.31]
// CHECK:STDOUT:     %.loc11_28.5: i32 = converted %.loc11_28.1, %.loc11_28.4 [template = constants.%.31]
// CHECK:STDOUT:     %int.make_type_signed.loc11_27: init type = call %Int.ref.loc11_24(%.loc11_28.5) [template = constants.%.32]
// CHECK:STDOUT:     %.loc11_30.1: type = value_of_initializer %int.make_type_signed.loc11_27 [template = constants.%.32]
// CHECK:STDOUT:     %.loc11_30.2: type = converted %int.make_type_signed.loc11_27, %.loc11_30.1 [template = constants.%.32]
// CHECK:STDOUT:     %n.param: %.32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSymbolic.decl: %UseSymbolic.type = fn_decl @UseSymbolic [template = constants.%UseSymbolic] {
// CHECK:STDOUT:     %n.patt: %.36 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.36 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.36 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.36 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc15_19: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc15_23.1: Core.IntLiteral = int_value 24 [template = constants.%.33]
// CHECK:STDOUT:     %.loc15_23.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc15_23.3: <bound method> = bound_method %.loc15_23.1, %.loc15_23.2 [template = constants.%.34]
// CHECK:STDOUT:     %int.convert_checked.loc15_23: init i32 = call %.loc15_23.3(%.loc15_23.1) [template = constants.%.35]
// CHECK:STDOUT:     %.loc15_23.4: i32 = value_of_initializer %int.convert_checked.loc15_23 [template = constants.%.35]
// CHECK:STDOUT:     %.loc15_23.5: i32 = converted %.loc15_23.1, %.loc15_23.4 [template = constants.%.35]
// CHECK:STDOUT:     %int.make_type_signed.loc15_22: init type = call %Int.ref.loc15_19(%.loc15_23.5) [template = constants.%.36]
// CHECK:STDOUT:     %.loc15_25.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%.36]
// CHECK:STDOUT:     %.loc15_25.2: type = converted %int.make_type_signed.loc15_22, %.loc15_25.1 [template = constants.%.36]
// CHECK:STDOUT:     %Int.ref.loc15_31: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc15_35.1: Core.IntLiteral = int_value 24 [template = constants.%.33]
// CHECK:STDOUT:     %.loc15_35.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc15_35.3: <bound method> = bound_method %.loc15_35.1, %.loc15_35.2 [template = constants.%.34]
// CHECK:STDOUT:     %int.convert_checked.loc15_35: init i32 = call %.loc15_35.3(%.loc15_35.1) [template = constants.%.35]
// CHECK:STDOUT:     %.loc15_35.4: i32 = value_of_initializer %int.convert_checked.loc15_35 [template = constants.%.35]
// CHECK:STDOUT:     %.loc15_35.5: i32 = converted %.loc15_35.1, %.loc15_35.4 [template = constants.%.35]
// CHECK:STDOUT:     %int.make_type_signed.loc15_34: init type = call %Int.ref.loc15_31(%.loc15_35.5) [template = constants.%.36]
// CHECK:STDOUT:     %.loc15_37.1: type = value_of_initializer %int.make_type_signed.loc15_34 [template = constants.%.36]
// CHECK:STDOUT:     %.loc15_37.2: type = converted %int.make_type_signed.loc15_34, %.loc15_37.1 [template = constants.%.36]
// CHECK:STDOUT:     %n.param: %.36 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.36 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.36 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseF(%n.param_patt: %.28) -> %.28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.2 [template = constants.%F]
// CHECK:STDOUT:   %n.ref: %.28 = name_ref n, %n
// CHECK:STDOUT:   %F.call: init %.28 = call %F.ref(%n.ref)
// CHECK:STDOUT:   %.loc8_14.1: %.28 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc8_14.2: %.28 = converted %F.call, %.loc8_14.1
// CHECK:STDOUT:   return %.loc8_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.28) -> %.28;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseG(%n.param_patt: %.32) -> %.32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%import_ref.3 [template = constants.%G]
// CHECK:STDOUT:   %n.ref: %.32 = name_ref n, %n
// CHECK:STDOUT:   %G.call: init %.32 = call %G.ref(%n.ref)
// CHECK:STDOUT:   %.loc12_14.1: %.32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc12_14.2: %.32 = converted %G.call, %.loc12_14.1
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.32) -> %.32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseSymbolic(%n.param_patt: %.36) -> %.36 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Symbolic.ref: %Symbolic.type = name_ref Symbolic, imports.%import_ref.4 [template = constants.%Symbolic]
// CHECK:STDOUT:   %.loc16_19: Core.IntLiteral = int_value 24 [template = constants.%.33]
// CHECK:STDOUT:   %n.ref: %.36 = name_ref n, %n
// CHECK:STDOUT:   %.loc16_18.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_18.2: <bound method> = bound_method %.loc16_19, %.loc16_18.1 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc16: init i32 = call %.loc16_18.2(%.loc16_19) [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_18.3: i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_18.4: i32 = converted %.loc16_19, %.loc16_18.3 [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_10: <specific function> = specific_function %Symbolic.ref, @Symbolic(constants.%.35) [template = constants.%.38]
// CHECK:STDOUT:   %Symbolic.call: init %.36 = call %.loc16_10(%n.ref)
// CHECK:STDOUT:   %.loc16_25.1: %.36 = value_of_initializer %Symbolic.call
// CHECK:STDOUT:   %.loc16_25.2: %.36 = converted %Symbolic.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(constants.%N.2: i32) {
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.37)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.1 (%.37)) -> @Symbolic.%.1 (%.37);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%.35) {
// CHECK:STDOUT:   %N => constants.%.35
// CHECK:STDOUT:   %N.patt => constants.%.35
// CHECK:STDOUT:   %.1 => constants.%.36
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:   %.loc10_12.1: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_12.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_12.3: <bound method> = bound_method %.loc10_12.1, %.loc10_12.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc10_12.3(%.loc10_12.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_12.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_12.5: i32 = converted %.loc10_12.1, %.loc10_12.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc10_12.5) [template = <error>]
// CHECK:STDOUT:   %.loc10_13.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc10_13.2: type = converted %int.make_type_signed, %.loc10_13.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.28: i32 = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref.1
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
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_19.1: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_19.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_19.3: <bound method> = bound_method %.loc12_19.1, %.loc12_19.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc12_19.3(%.loc12_19.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_19.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_19.5: i32 = converted %.loc12_19.1, %.loc12_19.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_19.5) [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_18.1: i32 = value_of_initializer %int.snegate [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_18.2: i32 = converted %int.snegate, %.loc12_18.1 [template = constants.%.28]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc12_18.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_signed, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:   %.loc9_12.1: Core.IntLiteral = int_value 1000000000 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_12.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc9_12.3: <bound method> = bound_method %.loc9_12.1, %.loc9_12.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc9_12.3(%.loc9_12.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_12.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_12.5: i32 = converted %.loc9_12.1, %.loc9_12.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc9_12.5) [template = <error>]
// CHECK:STDOUT:   %.loc9_22.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %int.make_type_signed, %.loc9_22.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
