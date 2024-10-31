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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
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
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 13 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %n.patt: %.3 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.3 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc6_9: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc6_13: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc6_12: init type = call %Int.ref.loc6_9(%.loc6_13) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_15.1: type = value_of_initializer %int.make_type_signed.loc6_12 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %int.make_type_signed.loc6_12, %.loc6_15.1 [template = constants.%.3]
// CHECK:STDOUT:     %Int.ref.loc6_21: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc6_25: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc6_24: init type = call %Int.ref.loc6_21(%.loc6_25) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_27.1: type = value_of_initializer %int.make_type_signed.loc6_24 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_27.2: type = converted %int.make_type_signed.loc6_24, %.loc6_27.1 [template = constants.%.3]
// CHECK:STDOUT:     %n.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.3 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %.5 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.5 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc10_9: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc10_13: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc10_12: init type = call %Int.ref.loc10_9(%.loc10_13) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_15.1: type = value_of_initializer %int.make_type_signed.loc10_12 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_15.2: type = converted %int.make_type_signed.loc10_12, %.loc10_15.1 [template = constants.%.5]
// CHECK:STDOUT:     %Int.ref.loc10_21: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc10_25: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc10_24: init type = call %Int.ref.loc10_21(%.loc10_25) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_24 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_24, %.loc10_27.1 [template = constants.%.5]
// CHECK:STDOUT:     %n.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.5 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%.loc14_28 (%.6) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%.loc14_28 (%.6) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%.loc14_28 (%.6) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%.loc14_28 (%.6) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %int.make_type_32, %.loc14_17.1 [template = i32]
// CHECK:STDOUT:     %Int.ref.loc14_25: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_29: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_28: init type = call %Int.ref.loc14_25(%N.ref.loc14_29) [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %.loc14_30.1: type = value_of_initializer %int.make_type_signed.loc14_28 [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %int.make_type_signed.loc14_28, %.loc14_30.1 [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %Int.ref.loc14_36: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_40: i32 = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_39: init type = call %Int.ref.loc14_36(%N.ref.loc14_40) [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %int.make_type_signed.loc14_39 [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %int.make_type_signed.loc14_39, %.loc14_41.1 [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Symbolic.%.loc14_28 (%.6) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%.loc14_28 (%.6) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%.loc14_28 (%.6) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%.loc14_28 (%.6) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.3) -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.5) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.5 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: i32) {
// CHECK:STDOUT:   %N.loc14_13.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   %.loc14_28: type = int_type signed, %N.loc14_13.2 [symbolic = %.loc14_28 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.loc14_28 (%.6)) -> @Symbolic.%.loc14_28 (%.6) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%.loc14_28 (%.6) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %.loc14_28 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %UseF.type: type = fn_type @UseF [template]
// CHECK:STDOUT:   %UseF: %UseF.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 13 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %UseG.type: type = fn_type @UseG [template]
// CHECK:STDOUT:   %UseG: %UseG.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 24 [template]
// CHECK:STDOUT:   %.7: type = int_type signed, %.6 [template]
// CHECK:STDOUT:   %UseSymbolic.type: type = fn_type @UseSymbolic [template]
// CHECK:STDOUT:   %UseSymbolic: %UseSymbolic.type = struct_value () [template]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.8: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.9: <specific function> = specific_function %Symbolic, @Symbolic(%.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %import_ref.2: %F.type = import_ref Main//use_types, inst+36, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.3: %G.type = import_ref Main//use_types, inst+61, loaded [template = constants.%G]
// CHECK:STDOUT:   %import_ref.4: %Symbolic.type = import_ref Main//use_types, inst+101, loaded [template = constants.%Symbolic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
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
// CHECK:STDOUT:     %n.patt: %.3 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.3 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc7_12: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc7_16: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc7_15: init type = call %Int.ref.loc7_12(%.loc7_16) [template = constants.%.3]
// CHECK:STDOUT:     %.loc7_18.1: type = value_of_initializer %int.make_type_signed.loc7_15 [template = constants.%.3]
// CHECK:STDOUT:     %.loc7_18.2: type = converted %int.make_type_signed.loc7_15, %.loc7_18.1 [template = constants.%.3]
// CHECK:STDOUT:     %Int.ref.loc7_24: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc7_28: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call %Int.ref.loc7_24(%.loc7_28) [template = constants.%.3]
// CHECK:STDOUT:     %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%.3]
// CHECK:STDOUT:     %.loc7_30.2: type = converted %int.make_type_signed.loc7_27, %.loc7_30.1 [template = constants.%.3]
// CHECK:STDOUT:     %n.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.3 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseG.decl: %UseG.type = fn_decl @UseG [template = constants.%UseG] {
// CHECK:STDOUT:     %n.patt: %.5 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.5 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc11_12: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc11_16: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc11_15: init type = call %Int.ref.loc11_12(%.loc11_16) [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %int.make_type_signed.loc11_15, %.loc11_18.1 [template = constants.%.5]
// CHECK:STDOUT:     %Int.ref.loc11_24: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc11_28: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc11_27: init type = call %Int.ref.loc11_24(%.loc11_28) [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_30.1: type = value_of_initializer %int.make_type_signed.loc11_27 [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_30.2: type = converted %int.make_type_signed.loc11_27, %.loc11_30.1 [template = constants.%.5]
// CHECK:STDOUT:     %n.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.5 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSymbolic.decl: %UseSymbolic.type = fn_decl @UseSymbolic [template = constants.%UseSymbolic] {
// CHECK:STDOUT:     %n.patt: %.7 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.7 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.7 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.7 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc15_19: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc15_23: i32 = int_literal 24 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_signed.loc15_22: init type = call %Int.ref.loc15_19(%.loc15_23) [template = constants.%.7]
// CHECK:STDOUT:     %.loc15_25.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%.7]
// CHECK:STDOUT:     %.loc15_25.2: type = converted %int.make_type_signed.loc15_22, %.loc15_25.1 [template = constants.%.7]
// CHECK:STDOUT:     %Int.ref.loc15_31: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %.loc15_35: i32 = int_literal 24 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_signed.loc15_34: init type = call %Int.ref.loc15_31(%.loc15_35) [template = constants.%.7]
// CHECK:STDOUT:     %.loc15_37.1: type = value_of_initializer %int.make_type_signed.loc15_34 [template = constants.%.7]
// CHECK:STDOUT:     %.loc15_37.2: type = converted %int.make_type_signed.loc15_34, %.loc15_37.1 [template = constants.%.7]
// CHECK:STDOUT:     %n.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.7 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.7 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseF(%n.param_patt: %.3) -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.2 [template = constants.%F]
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, %n
// CHECK:STDOUT:   %F.call: init %.3 = call %F.ref(%n.ref)
// CHECK:STDOUT:   %.loc8_14.1: %.3 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc8_14.2: %.3 = converted %F.call, %.loc8_14.1
// CHECK:STDOUT:   return %.loc8_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %.3) -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseG(%n.param_patt: %.5) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%import_ref.3 [template = constants.%G]
// CHECK:STDOUT:   %n.ref: %.5 = name_ref n, %n
// CHECK:STDOUT:   %G.call: init %.5 = call %G.ref(%n.ref)
// CHECK:STDOUT:   %.loc12_14.1: %.5 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc12_14.2: %.5 = converted %G.call, %.loc12_14.1
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %.5) -> %.5;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseSymbolic(%n.param_patt: %.7) -> %.7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Symbolic.ref: %Symbolic.type = name_ref Symbolic, imports.%import_ref.4 [template = constants.%Symbolic]
// CHECK:STDOUT:   %.loc16_19: i32 = int_literal 24 [template = constants.%.6]
// CHECK:STDOUT:   %n.ref: %.7 = name_ref n, %n
// CHECK:STDOUT:   %.loc16_10: <specific function> = specific_function %Symbolic.ref, @Symbolic(constants.%.6) [template = constants.%.9]
// CHECK:STDOUT:   %Symbolic.call: init %.7 = call %.loc16_10(%n.ref)
// CHECK:STDOUT:   %.loc16_25.1: %.7 = value_of_initializer %Symbolic.call
// CHECK:STDOUT:   %.loc16_25.2: %.7 = converted %Symbolic.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(constants.%N: i32) {
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %x.param_patt: @Symbolic.%.1 (%.8)) -> @Symbolic.%.1 (%.8);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%.6) {
// CHECK:STDOUT:   %N => constants.%.6
// CHECK:STDOUT:   %N.patt => constants.%.6
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref [template = constants.%Int]
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc10_12) [template = <error>]
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %.loc12_19: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_19) [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_18.1: i32 = value_of_initializer %int.snegate [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_18.2: i32 = converted %int.snegate, %.loc12_18.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc12_18.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_signed, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %Int.type = import_ref Main//types, inst+22, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref [template = constants.%Int]
// CHECK:STDOUT:   %.loc9_12: i32 = int_literal 1000000000 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc9_12) [template = <error>]
// CHECK:STDOUT:   %.loc9_22.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %int.make_type_signed, %.loc9_22.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
