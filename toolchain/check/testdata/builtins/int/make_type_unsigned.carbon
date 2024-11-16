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

fn IntLiteral() -> type = "int_literal.make_type";
fn UInt(n: IntLiteral()) -> type = "int.make_type_unsigned";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: UInt(64)) -> UInt(64) {
  return n;
}

fn G(n: UInt(13)) -> UInt(13) {
  return n;
}

fn Symbolic(N:! IntLiteral(), x: UInt(N)) -> UInt(N) {
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
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
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
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .UInt = %UInt.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.decl: %UInt.type = fn_decl @UInt [template = constants.%UInt] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_23.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_23.2: type = converted %int_literal.make_type, %.loc5_23.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.5: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Main//types, inst+7, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, inst+23, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .UInt = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Symbolic = %Symbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %n.patt: %.2 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.2 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc6_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_14: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_13: init type = call %UInt.ref.loc6_9(%.loc6_14) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_13 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_13, %.loc6_16.1 [template = constants.%.2]
// CHECK:STDOUT:     %UInt.ref.loc6_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc6_27: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_26: init type = call %UInt.ref.loc6_22(%.loc6_27) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_26 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_26, %.loc6_29.1 [template = constants.%.2]
// CHECK:STDOUT:     %n.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.2 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %.4 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.4 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc10_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_14: Core.IntLiteral = int_value 13 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_13: init type = call %UInt.ref.loc10_9(%.loc10_14) [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_13 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_13, %.loc10_16.1 [template = constants.%.4]
// CHECK:STDOUT:     %UInt.ref.loc10_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %.loc10_27: Core.IntLiteral = int_value 13 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_26: init type = call %UInt.ref.loc10_22(%.loc10_27) [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_26 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_26, %.loc10_29.1 [template = constants.%.4]
// CHECK:STDOUT:     %n.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.4 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%.loc14_38 (%.5) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%.loc14_38 (%.5) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%.loc14_38 (%.5) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%.loc14_38 (%.5) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.2: type = converted %int_literal.make_type, %.loc14_28.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %UInt.ref.loc14_34: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_39: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_38: init type = call %UInt.ref.loc14_34(%N.ref.loc14_39) [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %.loc14_40.1: type = value_of_initializer %int.make_type_unsigned.loc14_38 [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %.loc14_40.2: type = converted %int.make_type_unsigned.loc14_38, %.loc14_40.1 [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %UInt.ref.loc14_46: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_51: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_50: init type = call %UInt.ref.loc14_46(%N.ref.loc14_51) [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %.loc14_52.1: type = value_of_initializer %int.make_type_unsigned.loc14_50 [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %.loc14_52.2: type = converted %int.make_type_unsigned.loc14_50, %.loc14_52.1 [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Symbolic.%.loc14_38 (%.5) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%.loc14_38 (%.5) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%.loc14_38 (%.5) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%.loc14_38 (%.5) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.2) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.4) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.4 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc14_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   %.loc14_38: type = int_type unsigned, %N.loc14_13.2 [symbolic = %.loc14_38 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %x.param_patt: @Symbolic.%.loc14_38 (%.5)) -> @Symbolic.%.loc14_38 (%.5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%.loc14_38 (%.5) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %.loc14_38 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, inst+23, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .UInt = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:   %.loc10_13: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc10_13) [template = <error>]
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc10_14.2: type = converted %int.make_type_unsigned, %.loc10_14.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.28: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %.29: %.2 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, inst+23, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.3
// CHECK:STDOUT:     .ImplicitAs = %import_ref.4
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .UInt = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: %.2 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.2 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_14: init type = call constants.%Int(%.loc6_14.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6_14 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6_14, %.loc6_14.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_22: init type = call constants.%Int(%.loc6_22.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.2: type = value_of_initializer %int.make_type_signed.loc6_22 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.3: type = converted %int.make_type_signed.loc6_22, %.loc6_22.2 [template = constants.%.2]
// CHECK:STDOUT:     %n.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.2 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_20.1: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_20.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_20.3: <bound method> = bound_method %.loc12_20.1, %.loc12_20.2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc12_20.4: <specific function> = specific_function %.loc12_20.3, @Convert.2(constants.%.1) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc12_20: init %.2 = call %.loc12_20.4(%.loc12_20.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_20.5: %.2 = value_of_initializer %int.convert_checked.loc12_20 [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_20.6: %.2 = converted %.loc12_20.1, %.loc12_20.5 [template = constants.%.28]
// CHECK:STDOUT:   %int.snegate: init %.2 = call %Negate.ref(%.loc12_20.6) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_19.1: %Convert.type.6 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_19.2: <bound method> = bound_method %int.snegate, %.loc12_19.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_19.3: <specific function> = specific_function %.loc12_19.2, @Convert.4(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_19.4: %.2 = value_of_initializer %int.snegate [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_19.5: %.2 = converted %int.snegate, %.loc12_19.4 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc12_19: init Core.IntLiteral = call %.loc12_19.3(%.loc12_19.5) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_19.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc12_19 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_19.7: Core.IntLiteral = converted %int.snegate, %.loc12_19.6 [template = constants.%.33]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc12_19.7) [template = <error>]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_unsigned, %.loc12_22.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: %.2) -> %.2 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, inst+23, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .UInt = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:   %.loc9_13: Core.IntLiteral = int_value 1000000000 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc9_13) [template = <error>]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
