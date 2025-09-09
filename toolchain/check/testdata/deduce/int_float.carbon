// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/int_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/int_float.carbon

// --- int.carbon

library "[[@TEST_NAME]]";

fn F[N:! Core.IntLiteral()](n: Core.Int(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: i64) -> Core.IntLiteral() {
  return F(a);
}

// --- float.carbon

library "[[@TEST_NAME]]";

fn F[N:! Core.IntLiteral()](n: Core.Float(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: f64) -> Core.IntLiteral() {
  return F(a);
}

// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.896: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b4f: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %n.patt: @F.%pattern_type (%pattern_type.896) = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: @F.%pattern_type (%pattern_type.896) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_48: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_52: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc4_64: init type = call %IntLiteral.ref.loc4_52() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_64.1: type = value_of_initializer %IntLiteral.call.loc4_64 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_64.2: type = converted %IntLiteral.call.loc4_64, %.loc4_64.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call.loc4_26: init type = call %IntLiteral.ref.loc4_14() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %IntLiteral.call.loc4_26 [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %IntLiteral.call.loc4_26, %.loc4_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:     %n.param: @F.%Int.loc4_42.1 (%Int) = value_param call_param0
// CHECK:STDOUT:     %.loc4_42: type = splice_block %Int.loc4_42.2 [symbolic = %Int.loc4_42.1 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc4_32: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc4: Core.IntLiteral = name_ref N, %N.loc4_6.2 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc4_42.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc4_42.1 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @F.%Int.loc4_42.1 (%Int) = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.95b = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.95b = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %IntLiteral.call, %.loc8_33.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %i64 = value_param call_param0
// CHECK:STDOUT:     %.loc8_9: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:       %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_6.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:   %Int.loc4_42.1: type = class_type @Int, @Int(%N.loc4_6.1) [symbolic = %Int.loc4_42.1 (constants.%Int)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc4_42.1 [symbolic = %pattern_type (constants.%pattern_type.896)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc4_42.1 [symbolic = %require_complete (constants.%require_complete.b4f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%Int.loc4_42.1 (%Int)) -> Core.IntLiteral {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc5: Core.IntLiteral = name_ref N, %N.loc4_6.2 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %i64) -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: %i64 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_64) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init Core.IntLiteral = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   %.loc9_14.1: Core.IntLiteral = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_14.2: Core.IntLiteral = converted %F.call, %.loc9_14.1
// CHECK:STDOUT:   return %.loc9_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc4_6.1 => constants.%N
// CHECK:STDOUT:   %Int.loc4_42.1 => constants.%Int
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.896
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_64) {
// CHECK:STDOUT:   %N.loc4_6.1 => constants.%int_64
// CHECK:STDOUT:   %Int.loc4_42.1 => constants.%i64
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.95b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %Float: type = class_type @Float, @Float(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.f08: type = pattern_type %Float [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.957: <witness> = require_complete_type %Float [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %complete_type.7b9: <witness> = complete_type_witness %f64.794 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %n.patt: @F.%pattern_type (%pattern_type.f08) = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: @F.%pattern_type (%pattern_type.f08) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_54: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc4_66: init type = call %IntLiteral.ref.loc4_54() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_66.1: type = value_of_initializer %IntLiteral.call.loc4_66 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_66.2: type = converted %IntLiteral.call.loc4_66, %.loc4_66.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call.loc4_26: init type = call %IntLiteral.ref.loc4_14() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %IntLiteral.call.loc4_26 [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %IntLiteral.call.loc4_26, %.loc4_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:     %n.param: @F.%Float.loc4_44.1 (%Float) = value_param call_param0
// CHECK:STDOUT:     %.loc4_44: type = splice_block %Float.loc4_44.2 [symbolic = %Float.loc4_44.1 (constants.%Float)] {
// CHECK:STDOUT:       %Core.ref.loc4_32: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Float.ref: %Float.type = name_ref Float, imports.%Core.Float [concrete = constants.%Float.generic]
// CHECK:STDOUT:       %N.ref.loc4: Core.IntLiteral = name_ref N, %N.loc4_6.2 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:       %Float.loc4_44.2: type = class_type @Float, @Float(constants.%N) [symbolic = %Float.loc4_44.1 (constants.%Float)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @F.%Float.loc4_44.1 (%Float) = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.0ae = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %IntLiteral.call, %.loc8_33.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %f64.d77 = value_param call_param0
// CHECK:STDOUT:     %.loc8_9: type = splice_block %f64 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:       %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %f64.d77 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_6.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:   %Float.loc4_44.1: type = class_type @Float, @Float(%N.loc4_6.1) [symbolic = %Float.loc4_44.1 (constants.%Float)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Float.loc4_44.1 [symbolic = %pattern_type (constants.%pattern_type.f08)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Float.loc4_44.1 [symbolic = %require_complete (constants.%require_complete.957)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%Float.loc4_44.1 (%Float)) -> Core.IntLiteral {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc5: Core.IntLiteral = name_ref N, %N.loc4_6.2 [symbolic = %N.loc4_6.1 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %f64.d77) -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_64) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init Core.IntLiteral = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   %.loc9_14.1: Core.IntLiteral = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_14.2: Core.IntLiteral = converted %F.call, %.loc9_14.1
// CHECK:STDOUT:   return %.loc9_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc4_6.1 => constants.%N
// CHECK:STDOUT:   %Float.loc4_44.1 => constants.%Float
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_64) {
// CHECK:STDOUT:   %N.loc4_6.1 => constants.%int_64
// CHECK:STDOUT:   %Float.loc4_44.1 => constants.%f64.d77
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ae
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
