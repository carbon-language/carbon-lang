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

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(n: IntLiteral()) -> type = "int.make_type_signed";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: Int(64)) -> Int(64) {
  return n;
}

fn G(n: Int(13)) -> Int(13) {
  return n;
}

fn Symbolic(N:! IntLiteral(), x: Int(N)) -> Int(N) {
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
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
var n: Int(0);

// --- fail_negative_size.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: error: integer type width of -1 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: Int(Negate(1));
// CHECK:STDERR:        ^~~~~~~~~~~~~~
// CHECK:STDERR:
var n: Int(Negate(1));

// --- fail_oversized.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var m: Int(1000000000);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
var m: Int(1000000000);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
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
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int_literal.make_type, %.loc5_22.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %i13: type = int_type signed, %int_13 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Main//types, inst+7, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Main//types, inst+23, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .Int = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Symbolic = %Symbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %n.patt: %i64 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i64 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc6_9: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_64.loc6_13: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed.loc6_15: init type = call %Int.ref.loc6_9(%int_64.loc6_13) [template = constants.%i64]
// CHECK:STDOUT:     %.loc6_15.1: type = value_of_initializer %int.make_type_signed.loc6_15 [template = constants.%i64]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %int.make_type_signed.loc6_15, %.loc6_15.1 [template = constants.%i64]
// CHECK:STDOUT:     %Int.ref.loc6_21: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_64.loc6_25: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed.loc6_27: init type = call %Int.ref.loc6_21(%int_64.loc6_25) [template = constants.%i64]
// CHECK:STDOUT:     %.loc6_27.1: type = value_of_initializer %int.make_type_signed.loc6_27 [template = constants.%i64]
// CHECK:STDOUT:     %.loc6_27.2: type = converted %int.make_type_signed.loc6_27, %.loc6_27.1 [template = constants.%i64]
// CHECK:STDOUT:     %n.param: %i64 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i64 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %i13 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i13 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i13 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i13 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc10_9: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_13.loc10_13: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_signed.loc10_15: init type = call %Int.ref.loc10_9(%int_13.loc10_13) [template = constants.%i13]
// CHECK:STDOUT:     %.loc10_15.1: type = value_of_initializer %int.make_type_signed.loc10_15 [template = constants.%i13]
// CHECK:STDOUT:     %.loc10_15.2: type = converted %int.make_type_signed.loc10_15, %.loc10_15.1 [template = constants.%i13]
// CHECK:STDOUT:     %Int.ref.loc10_21: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_13.loc10_25: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_signed.loc10_27: init type = call %Int.ref.loc10_21(%int_13.loc10_25) [template = constants.%i13]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_27 [template = constants.%i13]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_27, %.loc10_27.1 [template = constants.%i13]
// CHECK:STDOUT:     %n.param: %i13 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i13 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i13 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i13 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%iN (%iN) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%iN (%iN) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%iN (%iN) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%iN (%iN) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.2: type = converted %int_literal.make_type, %.loc14_28.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Int.ref.loc14_34: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_38: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_39: init type = call %Int.ref.loc14_34(%N.ref.loc14_38) [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %.loc14_39.1: type = value_of_initializer %int.make_type_signed.loc14_39 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %.loc14_39.2: type = converted %int.make_type_signed.loc14_39, %.loc14_39.1 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %Int.ref.loc14_45: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc14_49: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed.loc14_50: init type = call %Int.ref.loc14_45(%N.ref.loc14_49) [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %.loc14_50.1: type = value_of_initializer %int.make_type_signed.loc14_50 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %.loc14_50.2: type = converted %int.make_type_signed.loc14_50, %.loc14_50.1 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Symbolic.%iN (%iN) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%iN (%iN) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%iN (%iN) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%iN (%iN) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %i64) -> %i64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i64 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %i13) -> %i13 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i13 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc14_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N.loc14_13.2 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %x.param_patt: @Symbolic.%iN (%iN)) -> @Symbolic.%iN (%iN) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%iN (%iN) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %UseF.type: type = fn_type @UseF [template]
// CHECK:STDOUT:   %UseF: %UseF.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %i13: type = int_type signed, %int_13 [template]
// CHECK:STDOUT:   %UseG.type: type = fn_type @UseG [template]
// CHECK:STDOUT:   %UseG: %UseG.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [template]
// CHECK:STDOUT:   %i24: type = int_type signed, %int_24 [template]
// CHECK:STDOUT:   %UseSymbolic.type: type = fn_type @UseSymbolic [template]
// CHECK:STDOUT:   %UseSymbolic: %UseSymbolic.type = struct_value () [template]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Symbolic.specific_fn: <specific function> = specific_function %Symbolic, @Symbolic(%int_24) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Main//types, inst+23, loaded [template = constants.%Int]
// CHECK:STDOUT:   %import_ref.3: %F.type = import_ref Main//use_types, inst+34, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.4: %G.type = import_ref Main//use_types, inst+59, loaded [template = constants.%G]
// CHECK:STDOUT:   %import_ref.5: %Symbolic.type = import_ref Main//use_types, inst+98, loaded [template = constants.%Symbolic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .Int = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.3
// CHECK:STDOUT:     .G = imports.%import_ref.4
// CHECK:STDOUT:     .Symbolic = imports.%import_ref.5
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UseF = %UseF.decl
// CHECK:STDOUT:     .UseG = %UseG.decl
// CHECK:STDOUT:     .UseSymbolic = %UseSymbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UseF.decl: %UseF.type = fn_decl @UseF [template = constants.%UseF] {
// CHECK:STDOUT:     %n.patt: %i64 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i64 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc7_12: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_64.loc7_16: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed.loc7_18: init type = call %Int.ref.loc7_12(%int_64.loc7_16) [template = constants.%i64]
// CHECK:STDOUT:     %.loc7_18.1: type = value_of_initializer %int.make_type_signed.loc7_18 [template = constants.%i64]
// CHECK:STDOUT:     %.loc7_18.2: type = converted %int.make_type_signed.loc7_18, %.loc7_18.1 [template = constants.%i64]
// CHECK:STDOUT:     %Int.ref.loc7_24: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_64.loc7_28: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed.loc7_30: init type = call %Int.ref.loc7_24(%int_64.loc7_28) [template = constants.%i64]
// CHECK:STDOUT:     %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7_30 [template = constants.%i64]
// CHECK:STDOUT:     %.loc7_30.2: type = converted %int.make_type_signed.loc7_30, %.loc7_30.1 [template = constants.%i64]
// CHECK:STDOUT:     %n.param: %i64 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i64 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseG.decl: %UseG.type = fn_decl @UseG [template = constants.%UseG] {
// CHECK:STDOUT:     %n.patt: %i13 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i13 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i13 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i13 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc11_12: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_13.loc11_16: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_signed.loc11_18: init type = call %Int.ref.loc11_12(%int_13.loc11_16) [template = constants.%i13]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %int.make_type_signed.loc11_18 [template = constants.%i13]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %int.make_type_signed.loc11_18, %.loc11_18.1 [template = constants.%i13]
// CHECK:STDOUT:     %Int.ref.loc11_24: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_13.loc11_28: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_signed.loc11_30: init type = call %Int.ref.loc11_24(%int_13.loc11_28) [template = constants.%i13]
// CHECK:STDOUT:     %.loc11_30.1: type = value_of_initializer %int.make_type_signed.loc11_30 [template = constants.%i13]
// CHECK:STDOUT:     %.loc11_30.2: type = converted %int.make_type_signed.loc11_30, %.loc11_30.1 [template = constants.%i13]
// CHECK:STDOUT:     %n.param: %i13 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i13 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i13 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i13 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSymbolic.decl: %UseSymbolic.type = fn_decl @UseSymbolic [template = constants.%UseSymbolic] {
// CHECK:STDOUT:     %n.patt: %i24 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i24 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i24 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i24 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc15_19: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_24.loc15_23: Core.IntLiteral = int_value 24 [template = constants.%int_24]
// CHECK:STDOUT:     %int.make_type_signed.loc15_25: init type = call %Int.ref.loc15_19(%int_24.loc15_23) [template = constants.%i24]
// CHECK:STDOUT:     %.loc15_25.1: type = value_of_initializer %int.make_type_signed.loc15_25 [template = constants.%i24]
// CHECK:STDOUT:     %.loc15_25.2: type = converted %int.make_type_signed.loc15_25, %.loc15_25.1 [template = constants.%i24]
// CHECK:STDOUT:     %Int.ref.loc15_31: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %int_24.loc15_35: Core.IntLiteral = int_value 24 [template = constants.%int_24]
// CHECK:STDOUT:     %int.make_type_signed.loc15_37: init type = call %Int.ref.loc15_31(%int_24.loc15_35) [template = constants.%i24]
// CHECK:STDOUT:     %.loc15_37.1: type = value_of_initializer %int.make_type_signed.loc15_37 [template = constants.%i24]
// CHECK:STDOUT:     %.loc15_37.2: type = converted %int.make_type_signed.loc15_37, %.loc15_37.1 [template = constants.%i24]
// CHECK:STDOUT:     %n.param: %i24 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i24 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i24 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i24 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseF(%n.param_patt: %i64) -> %i64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.3 [template = constants.%F]
// CHECK:STDOUT:   %n.ref: %i64 = name_ref n, %n
// CHECK:STDOUT:   %F.call: init %i64 = call %F.ref(%n.ref)
// CHECK:STDOUT:   %.loc8_14.1: %i64 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc8_14.2: %i64 = converted %F.call, %.loc8_14.1
// CHECK:STDOUT:   return %.loc8_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %i64) -> %i64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseG(%n.param_patt: %i13) -> %i13 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%import_ref.4 [template = constants.%G]
// CHECK:STDOUT:   %n.ref: %i13 = name_ref n, %n
// CHECK:STDOUT:   %G.call: init %i13 = call %G.ref(%n.ref)
// CHECK:STDOUT:   %.loc12_14.1: %i13 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc12_14.2: %i13 = converted %G.call, %.loc12_14.1
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %i13) -> %i13;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseSymbolic(%n.param_patt: %i24) -> %i24 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Symbolic.ref: %Symbolic.type = name_ref Symbolic, imports.%import_ref.5 [template = constants.%Symbolic]
// CHECK:STDOUT:   %int_24.loc16: Core.IntLiteral = int_value 24 [template = constants.%int_24]
// CHECK:STDOUT:   %n.ref: %i24 = name_ref n, %n
// CHECK:STDOUT:   %Symbolic.specific_fn: <specific function> = specific_function %Symbolic.ref, @Symbolic(constants.%int_24) [template = constants.%Symbolic.specific_fn]
// CHECK:STDOUT:   %Symbolic.call: init %i24 = call %Symbolic.specific_fn(%n.ref)
// CHECK:STDOUT:   %.loc16_25.1: %i24 = value_of_initializer %Symbolic.call
// CHECK:STDOUT:   %.loc16_25.2: %i24 = converted %Symbolic.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %x.param_patt: @Symbolic.%iN (%iN)) -> @Symbolic.%iN (%iN);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%int_24) {
// CHECK:STDOUT:   %N => constants.%int_24
// CHECK:STDOUT:   %N.patt => constants.%int_24
// CHECK:STDOUT:   %iN => constants.%i24
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Main//types, inst+23, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .Int = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%int_0) [template = <error>]
// CHECK:STDOUT:   %.loc10_13.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc10_13.2: type = converted %int.make_type_signed, %.loc10_13.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type.1: type = fn_type @Int.1 [template]
// CHECK:STDOUT:   %Int.1: %Int.type.1 = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type.2: type = fn_type @Int.2 [template]
// CHECK:STDOUT:   %Int.2: %Int.type.2 = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1.1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_-1.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %Int.type.2 = import_ref Main//types, inst+23, loaded [template = constants.%Int.2]
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
// CHECK:STDOUT:     .Int = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_14: init type = call constants.%Int.1(%int_32.loc6_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_signed.loc6_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_signed.loc6_14, %.loc6_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_22: init type = call constants.%Int.1(%int_32.loc6_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int.make_type_signed.loc6_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int.make_type_signed.loc6_22, %.loc6_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.ref: %Int.type.2 = name_ref Int, imports.%import_ref.2 [template = constants.%Int.2]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_19: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_19: <bound method> = bound_method %int_1, %impl.elem0.loc12_19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_19: <specific function> = specific_function %Convert.bound.loc12_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_19: init %i32 = call %Convert.specific_fn.loc12_19(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %int.convert_checked.loc12_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int_1, %.loc12_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc12_19.2) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_20: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc12_20: <bound method> = bound_method %int.snegate, %impl.elem0.loc12_20 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_20: <specific function> = specific_function %Convert.bound.loc12_20, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc12_20.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc12_20.2: %i32 = converted %int.snegate, %.loc12_20.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_20: init Core.IntLiteral = call %Convert.specific_fn.loc12_20(%.loc12_20.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc12_20 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_20.4: Core.IntLiteral = converted %int.snegate, %.loc12_20.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc12_20.4) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_signed, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.2(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Main//types, inst+23, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.1
// CHECK:STDOUT:     .Int = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%int_1000000000) [template = <error>]
// CHECK:STDOUT:   %.loc9_22.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %int.make_type_signed, %.loc9_22.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
