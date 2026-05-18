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
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn A` [RedeclRedef]
// CHECK:STDERR: fn A(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, unused m: i32) -> i32 { return n; }

fn B(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn B` [RedeclRedef]
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn B(n: i32, unused m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn C` [RedeclRedef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %A.type.a167e8.1: type = fn_type @A.loc15 [concrete]
// CHECK:STDOUT:   %A.f944da.1: %A.type.a167e8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.a167e8.2: type = fn_type @A.loc23 [concrete]
// CHECK:STDOUT:   %A.f944da.2: %A.type.a167e8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %B.type.fb7f47.1: type = fn_type @B.loc25 [concrete]
// CHECK:STDOUT:   %B.99baca.1: %B.type.fb7f47.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.fb7f47.2: type = fn_type @B.loc33 [concrete]
// CHECK:STDOUT:   %B.99baca.2: %B.type.fb7f47.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.04952c.1: type = fn_type @C.loc35 [concrete]
// CHECK:STDOUT:   %C.4dda5b.1: %C.type.04952c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.04952c.2: type = fn_type @C.loc43 [concrete]
// CHECK:STDOUT:   %C.4dda5b.2: %C.type.04952c.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc15
// CHECK:STDOUT:     .B = %B.decl.loc25
// CHECK:STDOUT:     .C = %C.decl.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc15: %A.type.a167e8.1 = fn_decl @A.loc15 [concrete = constants.%A.f944da.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc15_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32.loc15_25 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc15_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc23: %A.type.a167e8.2 = fn_decl @A.loc23 [concrete = constants.%A.f944da.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc23_32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc23_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %i32.loc23_32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc23_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc23_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc25: %B.type.fb7f47.1 = fn_decl @B.loc25 [concrete = constants.%B.99baca.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc25_32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc25_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc25: Core.Form = init_form %i32.loc25_32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc25_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc25_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc33: %B.type.fb7f47.2 = fn_decl @B.loc33 [concrete = constants.%B.99baca.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc33_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc33_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc33: Core.Form = init_form %i32.loc33_25 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc33_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc33_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc35: %C.type.04952c.1 = fn_decl @C.loc35 [concrete = constants.%C.4dda5b.1] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc35_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc35_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc35: Core.Form = init_form %i32.loc35_25 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc35_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc35_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc43: %C.type.04952c.2 = fn_decl @C.loc43 [concrete = constants.%C.4dda5b.2] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.c6b = at_binding_pattern m, %m.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc43_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc43_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc43: Core.Form = init_form %i32.loc43_25 [concrete = constants.%.70c]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc43_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc43_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc15(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc23(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc23_45.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_45.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc23_45.2(%n.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.loc25(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc25_45.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_45.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc25_45.2(%n.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.loc33(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc35(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc43(%n.param: %i32, %m.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
