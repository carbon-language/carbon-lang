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
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var n: UInt(0);

// --- fail_negative_size.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: error: integer type width of -1 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var n: UInt(Negate(1));

// --- fail_oversized.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var m: UInt(1000000000);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %u64: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %u13: type = int_type unsigned, %int_13 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
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
// CHECK:STDOUT:     %n.patt: %u64 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %u64 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc6_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_64.loc6_14: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_16: init type = call %UInt.ref.loc6_9(%int_64.loc6_14) [template = constants.%u64]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_16 [template = constants.%u64]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_16, %.loc6_16.1 [template = constants.%u64]
// CHECK:STDOUT:     %UInt.ref.loc6_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_64.loc6_27: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_29: init type = call %UInt.ref.loc6_22(%int_64.loc6_27) [template = constants.%u64]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_29 [template = constants.%u64]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_29, %.loc6_29.1 [template = constants.%u64]
// CHECK:STDOUT:     %n.param: %u64 = value_param runtime_param0
// CHECK:STDOUT:     %n: %u64 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %u64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %u13 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %u13 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u13 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u13 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc10_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_13.loc10_14: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_16: init type = call %UInt.ref.loc10_9(%int_13.loc10_14) [template = constants.%u13]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_16 [template = constants.%u13]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_16, %.loc10_16.1 [template = constants.%u13]
// CHECK:STDOUT:     %UInt.ref.loc10_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_13.loc10_27: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_29: init type = call %UInt.ref.loc10_22(%int_13.loc10_27) [template = constants.%u13]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_29 [template = constants.%u13]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_29, %.loc10_29.1 [template = constants.%u13]
// CHECK:STDOUT:     %n.param: %u13 = value_param runtime_param0
// CHECK:STDOUT:     %n: %u13 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %u13 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u13 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%uN (%uN) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%uN (%uN) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%uN (%uN) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%uN (%uN) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.2: type = converted %int_literal.make_type, %.loc14_28.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %UInt.ref.loc14_34: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_39: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_40: init type = call %UInt.ref.loc14_34(%N.ref.loc14_39) [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %.loc14_40.1: type = value_of_initializer %int.make_type_unsigned.loc14_40 [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %.loc14_40.2: type = converted %int.make_type_unsigned.loc14_40, %.loc14_40.1 [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %UInt.ref.loc14_46: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_51: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_52: init type = call %UInt.ref.loc14_46(%N.ref.loc14_51) [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %.loc14_52.1: type = value_of_initializer %int.make_type_unsigned.loc14_52 [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %.loc14_52.2: type = converted %int.make_type_unsigned.loc14_52, %.loc14_52.1 [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Symbolic.%uN (%uN) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%uN (%uN) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%uN (%uN) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%uN (%uN) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %u64) -> %u64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %u64 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %u13) -> %u13 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %u13 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc14_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N.loc14_13.2 [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %x.param_patt: @Symbolic.%uN (%uN)) -> @Symbolic.%uN (%uN) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%uN (%uN) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%int_0) [template = <error>]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
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
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_14: init type = call constants.%Int(%int_32.loc6_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_signed.loc6_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_signed.loc6_14, %.loc6_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_22: init type = call constants.%Int(%int_32.loc6_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int.make_type_signed.loc6_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int.make_type_signed.loc6_22, %.loc6_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_20: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_20: <bound method> = bound_method %int_1, %impl.elem0.loc12_20 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_20: <specific function> = specific_function %Convert.bound.loc12_20, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_20: init %i32 = call %Convert.specific_fn.loc12_20(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_20.1: %i32 = value_of_initializer %int.convert_checked.loc12_20 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_20.2: %i32 = converted %int_1, %.loc12_20.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc12_20.2) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_21: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc12_21: <bound method> = bound_method %int.snegate, %impl.elem0.loc12_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_21: <specific function> = specific_function %Convert.bound.loc12_21, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int.snegate, %.loc12_21.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_21: init Core.IntLiteral = call %Convert.specific_fn.loc12_21(%.loc12_21.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_21.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc12_21 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_21.4: Core.IntLiteral = converted %int.snegate, %.loc12_21.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc12_21.4) [template = <error>]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_unsigned, %.loc12_22.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template]
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
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%int_1000000000) [template = <error>]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
