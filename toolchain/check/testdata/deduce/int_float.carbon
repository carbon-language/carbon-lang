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

fn F[N:! Core.IntLiteral()](unused n: Core.Int(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: i64) -> Core.IntLiteral() {
  return F(a);
}

// --- float.carbon

library "[[@TEST_NAME]]";

fn F[N:! Core.IntLiteral()](unused n: Core.Float(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: f64) -> Core.IntLiteral() {
  return F(a);
}

// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.142: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4df: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6f1: <witness> = impl_witness imports.%Copy.impl_witness_table.5e0 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.6f1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c52: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.c22: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c52, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.3c6: <bound method> = bound_method %N, %Core.IntLiteral.as.Copy.impl.Op [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.03c: <bound method> = bound_method %int_64, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.5e0 = impl_witness_table (%Core.import_ref.9c3), @Core.IntLiteral.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %n.param_patt: @F.%pattern_type (%pattern_type.142) = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: @F.%pattern_type (%pattern_type.142) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc4_71.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_55: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_59: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc4_71: init type = call %IntLiteral.ref.loc4_59() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.1: type = value_of_initializer %IntLiteral.call.loc4_71 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.2: type = converted %IntLiteral.call.loc4_71, %.loc4_71.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.3: Core.Form = init_form %.loc4_71.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call.loc4_26: init type = call %IntLiteral.ref.loc4_14() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %IntLiteral.call.loc4_26 [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %IntLiteral.call.loc4_26, %.loc4_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     %n.param: @F.%Int.loc4_49.1 (%Int) = value_param call_param0
// CHECK:STDOUT:     %.loc4_49: type = splice_block %Int.loc4_49.2 [symbolic = %Int.loc4_49.1 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc4: Core.IntLiteral = name_ref N, %N.loc4_7.2 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc4_49.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc4_49.1 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @F.%Int.loc4_49.1 (%Int) = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.f57 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.f57 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc8_33.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %IntLiteral.call, %.loc8_33.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.3: Core.Form = init_form %.loc8_33.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %a.param: %i64 = value_param call_param0
// CHECK:STDOUT:     %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:     %a: %i64 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:   %Int.loc4_49.1: type = class_type @Int, @Int(%N.loc4_7.1) [symbolic = %Int.loc4_49.1 (constants.%Int)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc4_49.1 [symbolic = %pattern_type (constants.%pattern_type.142)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc4_49.1 [symbolic = %require_complete (constants.%require_complete.4df)]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc4_7.1, constants.%Core.IntLiteral.as.Copy.impl.Op [symbolic = %Core.IntLiteral.as.Copy.impl.Op.bound (constants.%Core.IntLiteral.as.Copy.impl.Op.bound.3c6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%Int.loc4_49.1 (%Int)) -> out %return.param: Core.IntLiteral {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc5: Core.IntLiteral = name_ref N, %N.loc4_7.2 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.c22 = impl_witness_access constants.%Copy.impl_witness.6f1, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %N.ref.loc5, %impl.elem0 [symbolic = %Core.IntLiteral.as.Copy.impl.Op.bound (constants.%Core.IntLiteral.as.Copy.impl.Op.bound.3c6)]
// CHECK:STDOUT:     %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method(%N.ref.loc5) [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     return %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %i64) -> out %return.param: Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: %i64 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_64) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init Core.IntLiteral = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc4_7.1 => constants.%N
// CHECK:STDOUT:   %Int.loc4_49.1 => constants.%Int
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_64) {
// CHECK:STDOUT:   %N.loc4_7.1 => constants.%int_64
// CHECK:STDOUT:   %Int.loc4_49.1 => constants.%i64
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f57
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound => constants.%Core.IntLiteral.as.Copy.impl.Op.bound.03c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %Float: type = class_type @Float, @Float(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.903: type = pattern_type %Float [symbolic]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.eb0: <witness> = require_complete_type %Float [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6f1: <witness> = impl_witness imports.%Copy.impl_witness_table.5e0 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.6f1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c52: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.c22: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c52, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.3c6: <bound method> = bound_method %N, %Core.IntLiteral.as.Copy.impl.Op [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %complete_type.7b9: <witness> = complete_type_witness %f64.794 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.03c: <bound method> = bound_method %int_64, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.5e0 = impl_witness_table (%Core.import_ref.9c3), @Core.IntLiteral.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %n.param_patt: @F.%pattern_type (%pattern_type.903) = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: @F.%pattern_type (%pattern_type.903) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc4_73.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_57: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_61: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc4_73: init type = call %IntLiteral.ref.loc4_61() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_73.1: type = value_of_initializer %IntLiteral.call.loc4_73 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_73.2: type = converted %IntLiteral.call.loc4_73, %.loc4_73.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_73.3: Core.Form = init_form %.loc4_73.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call.loc4_26: init type = call %IntLiteral.ref.loc4_14() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %IntLiteral.call.loc4_26 [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %IntLiteral.call.loc4_26, %.loc4_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     %n.param: @F.%Float.loc4_51.1 (%Float) = value_param call_param0
// CHECK:STDOUT:     %.loc4_51: type = splice_block %Float.loc4_51.2 [symbolic = %Float.loc4_51.1 (constants.%Float)] {
// CHECK:STDOUT:       %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Float.ref: %Float.type = name_ref Float, imports.%Core.Float [concrete = constants.%Float.generic]
// CHECK:STDOUT:       %N.ref.loc4: Core.IntLiteral = name_ref N, %N.loc4_7.2 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:       %Float.loc4_51.2: type = class_type @Float, @Float(constants.%N) [symbolic = %Float.loc4_51.1 (constants.%Float)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @F.%Float.loc4_51.1 (%Float) = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.fb7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.fb7 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc8_33.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %IntLiteral.call, %.loc8_33.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_33.3: Core.Form = init_form %.loc8_33.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %a.param: %f64.dc1 = value_param call_param0
// CHECK:STDOUT:     %f64: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %a: %f64.dc1 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:   %Float.loc4_51.1: type = class_type @Float, @Float(%N.loc4_7.1) [symbolic = %Float.loc4_51.1 (constants.%Float)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Float.loc4_51.1 [symbolic = %pattern_type (constants.%pattern_type.903)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Float.loc4_51.1 [symbolic = %require_complete (constants.%require_complete.eb0)]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc4_7.1, constants.%Core.IntLiteral.as.Copy.impl.Op [symbolic = %Core.IntLiteral.as.Copy.impl.Op.bound (constants.%Core.IntLiteral.as.Copy.impl.Op.bound.3c6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%Float.loc4_51.1 (%Float)) -> out %return.param: Core.IntLiteral {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc5: Core.IntLiteral = name_ref N, %N.loc4_7.2 [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.c22 = impl_witness_access constants.%Copy.impl_witness.6f1, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %N.ref.loc5, %impl.elem0 [symbolic = %Core.IntLiteral.as.Copy.impl.Op.bound (constants.%Core.IntLiteral.as.Copy.impl.Op.bound.3c6)]
// CHECK:STDOUT:     %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method(%N.ref.loc5) [symbolic = %N.loc4_7.1 (constants.%N)]
// CHECK:STDOUT:     return %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %f64.dc1) -> out %return.param: Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: %f64.dc1 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_64) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init Core.IntLiteral = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc4_7.1 => constants.%N
// CHECK:STDOUT:   %Float.loc4_51.1 => constants.%Float
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_64) {
// CHECK:STDOUT:   %N.loc4_7.1 => constants.%int_64
// CHECK:STDOUT:   %Float.loc4_51.1 => constants.%f64.dc1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7b9
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound => constants.%Core.IntLiteral.as.Copy.impl.Op.bound.03c
// CHECK:STDOUT: }
// CHECK:STDOUT:
