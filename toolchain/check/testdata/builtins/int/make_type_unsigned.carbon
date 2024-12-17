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

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+29]]:8: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+25]]:13: error: cannot implicitly convert from `i32` to `Core.Int(N)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:             ^~~~~~~~~
// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+22]]:13: note: type `i32` does not implement interface `Core.ImplicitAs(Core.Int(N))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:             ^~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:21:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR: fail_negative_size.carbon:[[@LINE-17]]:1: in import [InImport]
// CHECK:STDERR: types.carbon:5:9: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn UInt(n: IntLiteral()) -> type = "int.make_type_unsigned";
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+11]]:13: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:             ^~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:21:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_size.carbon:[[@LINE-28]]:1: in import [InImport]
// CHECK:STDERR: types.carbon:5:9: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn UInt(n: IntLiteral()) -> type = "int.make_type_unsigned";
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [template]
// CHECK:STDOUT:   %u13.builtin: type = int_type unsigned, %int_13 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %uN.builtin: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %Symbolic.type: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %Symbolic: %Symbolic.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %uN.builtin [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Main//types, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, UInt, loaded [template = constants.%UInt]
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
// CHECK:STDOUT:     %n.patt: %u64.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %u64.builtin = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u64.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u64.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc6_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_64.loc6_14: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_16: init type = call %UInt.ref.loc6_9(%int_64.loc6_14) [template = constants.%u64.builtin]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_16 [template = constants.%u64.builtin]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_16, %.loc6_16.1 [template = constants.%u64.builtin]
// CHECK:STDOUT:     %UInt.ref.loc6_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_64.loc6_27: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_29: init type = call %UInt.ref.loc6_22(%int_64.loc6_27) [template = constants.%u64.builtin]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_29 [template = constants.%u64.builtin]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_29, %.loc6_29.1 [template = constants.%u64.builtin]
// CHECK:STDOUT:     %n.param: %u64.builtin = value_param runtime_param0
// CHECK:STDOUT:     %n: %u64.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %u64.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u64.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %u13.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %u13.builtin = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u13.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u13.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %UInt.ref.loc10_9: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_13.loc10_14: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_16: init type = call %UInt.ref.loc10_9(%int_13.loc10_14) [template = constants.%u13.builtin]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_16 [template = constants.%u13.builtin]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_16, %.loc10_16.1 [template = constants.%u13.builtin]
// CHECK:STDOUT:     %UInt.ref.loc10_22: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %int_13.loc10_27: Core.IntLiteral = int_value 13 [template = constants.%int_13]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_29: init type = call %UInt.ref.loc10_22(%int_13.loc10_27) [template = constants.%u13.builtin]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_29 [template = constants.%u13.builtin]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_29, %.loc10_29.1 [template = constants.%u13.builtin]
// CHECK:STDOUT:     %n.param: %u13.builtin = value_param runtime_param0
// CHECK:STDOUT:     %n: %u13.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %u13.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u13.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: %Symbolic.type = fn_decl @Symbolic [template = constants.%Symbolic] {
// CHECK:STDOUT:     %N.patt.loc14_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc14_13.1, runtime_param<invalid> [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Symbolic.%uN.builtin (%uN.builtin) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Symbolic.%uN.builtin (%uN.builtin) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Symbolic.%uN.builtin (%uN.builtin) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Symbolic.%uN.builtin (%uN.builtin) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_28.2: type = converted %int_literal.make_type, %.loc14_28.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %UInt.ref.loc14_34: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_39: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_40: init type = call %UInt.ref.loc14_34(%N.ref.loc14_39) [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %.loc14_40.1: type = value_of_initializer %int.make_type_unsigned.loc14_40 [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %.loc14_40.2: type = converted %int.make_type_unsigned.loc14_40, %.loc14_40.1 [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %UInt.ref.loc14_46: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:     %N.ref.loc14_51: Core.IntLiteral = name_ref N, %N.loc14_13.1 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_52: init type = call %UInt.ref.loc14_46(%N.ref.loc14_51) [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %.loc14_52.1: type = value_of_initializer %int.make_type_unsigned.loc14_52 [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %.loc14_52.2: type = converted %int.make_type_unsigned.loc14_52, %.loc14_52.1 [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_13.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Symbolic.%uN.builtin (%uN.builtin) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Symbolic.%uN.builtin (%uN.builtin) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Symbolic.%uN.builtin (%uN.builtin) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Symbolic.%uN.builtin (%uN.builtin) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %u64.builtin) -> %u64.builtin {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %u64.builtin = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %u13.builtin) -> %u13.builtin {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %u13.builtin = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Symbolic(%N.loc14_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc14_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   %uN.builtin: type = int_type unsigned, %N.loc14_13.2 [symbolic = %uN.builtin (constants.%uN.builtin)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Symbolic.%uN.builtin (%uN.builtin) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral, %x.param_patt: @Symbolic.%uN.builtin (%uN.builtin)) -> @Symbolic.%uN.builtin (%uN.builtin) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Symbolic.%uN.builtin (%uN.builtin) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Symbolic(constants.%N) {
// CHECK:STDOUT:   %N.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_13.2 => constants.%N
// CHECK:STDOUT:   %uN.builtin => constants.%uN.builtin
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
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, UInt, loaded [template = constants.%UInt]
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
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_1) [symbolic]
// CHECK:STDOUT:   %impl_witness.6: <witness> = impl_witness (%Convert.4), @impl.2(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, UInt, loaded [template = constants.%UInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.3
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
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
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.ref: %UInt.type = name_ref UInt, imports.%import_ref.2 [template = constants.%UInt]
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc37_20: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc37_20: <bound method> = bound_method %int_1, %impl.elem0.loc37_20 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc37_20: <specific function> = specific_function %Convert.bound.loc37_20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc37_20: init %Int = call %Convert.specific_fn.loc37_20(%int_1) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc37_20.1: %i32 = value_of_initializer %int.convert_checked.loc37_20 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc37_20.2: %i32 = converted %int_1, %.loc37_20.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc37_20.2)
// CHECK:STDOUT:   %impl.elem0.loc37_21: %Convert.type.5 = impl_witness_access constants.%impl_witness.6, element0 [symbolic = constants.%Convert.4]
// CHECK:STDOUT:   %Convert.bound.loc37_21: <bound method> = bound_method %int.snegate, %impl.elem0.loc37_21
// CHECK:STDOUT:   %Convert.specific_fn.loc37_21: <specific function> = specific_function %Convert.bound.loc37_21, @Convert.3(constants.%N)
// CHECK:STDOUT:   %.loc37_21.1: %Int = converted %int.snegate, <error> [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc37_21: init Core.IntLiteral = call %Convert.specific_fn.loc37_21(<error>)
// CHECK:STDOUT:   %.loc37_21.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc37_21
// CHECK:STDOUT:   %.loc37_21.3: Core.IntLiteral = converted %int.snegate, %.loc37_21.2
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc37_21.3)
// CHECK:STDOUT:   %.loc37_22.1: type = value_of_initializer %int.make_type_unsigned
// CHECK:STDOUT:   %.loc37_22.2: type = converted %int.make_type_unsigned, %.loc37_22.1
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%n.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned" [from "types.carbon"];
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
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//types, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.2: %UInt.type = import_ref Main//types, UInt, loaded [template = constants.%UInt]
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
// CHECK:STDOUT: fn @UInt(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_unsigned" [from "types.carbon"];
// CHECK:STDOUT:
