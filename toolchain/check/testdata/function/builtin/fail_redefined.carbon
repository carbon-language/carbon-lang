// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon

fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        ╭─┤ fail_redefined.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        · ──────────────┬──────────────
// CHECK:STDERR:        ·               ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn A(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: redefinition of `fn A` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn A(n: i32, unused m: i32) -> i32 { return n; }

fn B(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR:        ╭─┤ fail_redefined.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ fn B(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR:        · ──────────────────┬─────────────────
// CHECK:STDERR:        ·                   ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: redefinition of `fn B` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        ╭─┤ fail_redefined.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        · ──────────────┬──────────────
// CHECK:STDERR:        ·               ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: redefinition of `fn C` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %A.type.9f4a6d.1: type = fn_type @A.loc15 [concrete]
// CHECK:STDOUT:   %A.921eb9.1: %A.type.9f4a6d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.9f4a6d.2: type = fn_type @A.loc26 [concrete]
// CHECK:STDOUT:   %A.921eb9.2: %A.type.9f4a6d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %B.type.573c71.1: type = fn_type @B.loc28 [concrete]
// CHECK:STDOUT:   %B.08ed21.1: %B.type.573c71.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.573c71.2: type = fn_type @B.loc39 [concrete]
// CHECK:STDOUT:   %B.08ed21.2: %B.type.573c71.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.322e3b.1: type = fn_type @C.loc41 [concrete]
// CHECK:STDOUT:   %C.010386.1: %C.type.322e3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.322e3b.2: type = fn_type @C.loc52 [concrete]
// CHECK:STDOUT:   %C.010386.2: %C.type.322e3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc15
// CHECK:STDOUT:     .B = %B.decl.loc28
// CHECK:STDOUT:     .C = %C.decl.loc41
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc15: %A.type.9f4a6d.1 = fn_decl @A.loc15 [concrete = constants.%A.921eb9.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc15_25 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32.loc15_25 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc15_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc26: %A.type.9f4a6d.2 = fn_decl @A.loc26 [concrete = constants.%A.921eb9.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc26_32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc26_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc26: Core.Form = init_form %i32.loc26_32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc26_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc26_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc28: %B.type.573c71.1 = fn_decl @B.loc28 [concrete = constants.%B.08ed21.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc28_32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc28_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc28: Core.Form = init_form %i32.loc28_32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc28_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc28_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc39: %B.type.573c71.2 = fn_decl @B.loc39 [concrete = constants.%B.08ed21.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc39_25 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc39_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc39: Core.Form = init_form %i32.loc39_25 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc39_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc39_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc41: %C.type.322e3b.1 = fn_decl @C.loc41 [concrete = constants.%C.010386.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc41_25 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc41_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc41: Core.Form = init_form %i32.loc41_25 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc41_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc41_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc52: %C.type.322e3b.2 = fn_decl @C.loc52 [concrete = constants.%C.010386.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%m.param_patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.d94 = wrapper_binding_pattern m, %m.param_patt [concrete = constants.%m.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc52_25 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc52_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc52: Core.Form = init_form %i32.loc52_25 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc52_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc52_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = wrapper_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc15(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc26(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc26_45.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_45.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc26_45.2(%n.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.loc28(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc28_45.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_45.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc28_45.2(%n.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.loc39(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc41(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc52(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
