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
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 64 [template]
// CHECK:STDOUT:   %.38: type = int_type unsigned, %.37 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.15 [template]
// CHECK:STDOUT:   %.41: i32 = int_value 13 [template]
// CHECK:STDOUT:   %.42: type = int_type unsigned, %.41 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N.2: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.43: type = int_type unsigned, %N.2 [symbolic]
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
// CHECK:STDOUT:     %n.patt: %.38 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.38 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.38 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.38 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc6_9: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_14.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_14.3: <bound method> = bound_method %.loc6_14.1, %.loc6_14.2 [template = constants.%.36]
// CHECK:STDOUT:     %int.convert_checked.loc6_14: init i32 = call %.loc6_14.3(%.loc6_14.1) [template = constants.%.37]
// CHECK:STDOUT:     %.loc6_14.4: i32 = value_of_initializer %int.convert_checked.loc6_14 [template = constants.%.37]
// CHECK:STDOUT:     %.loc6_14.5: i32 = converted %.loc6_14.1, %.loc6_14.4 [template = constants.%.37]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_13: init type = call %UInt.ref.loc6_9(%.loc6_14.5) [template = constants.%.38]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_13 [template = constants.%.38]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_13, %.loc6_16.1 [template = constants.%.38]
// CHECK:STDOUT:     %UInt.ref.loc6_22: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_27.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_27.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc6_27.3: <bound method> = bound_method %.loc6_27.1, %.loc6_27.2 [template = constants.%.36]
// CHECK:STDOUT:     %int.convert_checked.loc6_27: init i32 = call %.loc6_27.3(%.loc6_27.1) [template = constants.%.37]
// CHECK:STDOUT:     %.loc6_27.4: i32 = value_of_initializer %int.convert_checked.loc6_27 [template = constants.%.37]
// CHECK:STDOUT:     %.loc6_27.5: i32 = converted %.loc6_27.1, %.loc6_27.4 [template = constants.%.37]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_26: init type = call %UInt.ref.loc6_22(%.loc6_27.5) [template = constants.%.38]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_26 [template = constants.%.38]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_26, %.loc6_29.1 [template = constants.%.38]
// CHECK:STDOUT:     %n.param: %.38 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.38 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.38 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.38 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %.42 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.42 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.42 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.42 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc10_9: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_14.1: Core.IntLiteral = int_value 13 [template = constants.%.39]
// CHECK:STDOUT:     %.loc10_14.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_14.3: <bound method> = bound_method %.loc10_14.1, %.loc10_14.2 [template = constants.%.40]
// CHECK:STDOUT:     %int.convert_checked.loc10_14: init i32 = call %.loc10_14.3(%.loc10_14.1) [template = constants.%.41]
// CHECK:STDOUT:     %.loc10_14.4: i32 = value_of_initializer %int.convert_checked.loc10_14 [template = constants.%.41]
// CHECK:STDOUT:     %.loc10_14.5: i32 = converted %.loc10_14.1, %.loc10_14.4 [template = constants.%.41]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_13: init type = call %UInt.ref.loc10_9(%.loc10_14.5) [template = constants.%.42]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_13 [template = constants.%.42]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_13, %.loc10_16.1 [template = constants.%.42]
// CHECK:STDOUT:     %UInt.ref.loc10_22: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_27.1: Core.IntLiteral = int_value 13 [template = constants.%.39]
// CHECK:STDOUT:     %.loc10_27.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc10_27.3: <bound method> = bound_method %.loc10_27.1, %.loc10_27.2 [template = constants.%.40]
// CHECK:STDOUT:     %int.convert_checked.loc10_27: init i32 = call %.loc10_27.3(%.loc10_27.1) [template = constants.%.41]
// CHECK:STDOUT:     %.loc10_27.4: i32 = value_of_initializer %int.convert_checked.loc10_27 [template = constants.%.41]
// CHECK:STDOUT:     %.loc10_27.5: i32 = converted %.loc10_27.1, %.loc10_27.4 [template = constants.%.41]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_26: init type = call %UInt.ref.loc10_22(%.loc10_27.5) [template = constants.%.42]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_26 [template = constants.%.42]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_26, %.loc10_29.1 [template = constants.%.42]
// CHECK:STDOUT:     %n.param: %.42 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.42 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.42 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.42 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%.loc14_29 (%.43) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%.loc14_29 (%.43) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%.loc14_29 (%.43) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%.loc14_29 (%.43) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %int.make_type_32, %.loc14_17.1 [template = i32]
// CHECK:STDOUT:     %UInt.ref.loc14_25: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_30: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_29: init type = call %UInt.ref.loc14_25(%N.ref.loc14_30) [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %.loc14_31.1: type = value_of_initializer %int.make_type_unsigned.loc14_29 [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %.loc14_31.2: type = converted %int.make_type_unsigned.loc14_29, %.loc14_31.1 [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %UInt.ref.loc14_37: %UInt.type = name_ref UInt, imports.%import_ref.1 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_42: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_41: init type = call %UInt.ref.loc14_37(%N.ref.loc14_42) [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %.loc14_43.1: type = value_of_initializer %int.make_type_unsigned.loc14_41 [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %.loc14_43.2: type = converted %int.make_type_unsigned.loc14_41, %.loc14_43.1 [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:     %x.param: @Symbolic.%.loc14_29 (%.43) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%.loc14_29 (%.43) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%.loc14_29 (%.43) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%.loc14_29 (%.43) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.38) -> %.38 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.38 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.42) -> %.42 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.42 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: i32) {
// CHECK:STDOUT:   %N.loc14_13.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %.loc14_29: type = int_type unsigned, %N.loc14_13.2 [symbolic = %.loc14_29 (constants.%.43)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.loc14_29 (%.43)) -> @Symbolic.%.loc14_29 (%.43) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%.loc14_29 (%.43) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N.2) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N.2
// CHECK:STDOUT:   %.loc14_29 => constants.%.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc10_13.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_13.3: <bound method> = bound_method %.loc10_13.1, %.loc10_13.2 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc10_13.3(%.loc10_13.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc10_13.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.37]
// CHECK:STDOUT:   %.loc10_13.5: i32 = converted %.loc10_13.1, %.loc10_13.4 [template = constants.%.37]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc10_13.5) [template = <error>]
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc10_14.2: type = converted %int.make_type_unsigned, %.loc10_14.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.38: i32 = int_value -1 [template]
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
// CHECK:STDOUT:   %.loc12_20.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_20.3: <bound method> = bound_method %.loc12_20.1, %.loc12_20.2 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc12_20.3(%.loc12_20.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_20.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_20.5: i32 = converted %.loc12_20.1, %.loc12_20.4 [template = constants.%.37]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_20.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc12_19.1: i32 = value_of_initializer %int.snegate [template = constants.%.38]
// CHECK:STDOUT:   %.loc12_19.2: i32 = converted %int.snegate, %.loc12_19.1 [template = constants.%.38]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc12_19.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_unsigned, %.loc12_22.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %UInt.type = import_ref Main//types, inst+22, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc9_13.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc9_13.3: <bound method> = bound_method %.loc9_13.1, %.loc9_13.2 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc9_13.3(%.loc9_13.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.5: i32 = converted %.loc9_13.1, %.loc9_13.4 [template = constants.%.37]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc9_13.5) [template = <error>]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
