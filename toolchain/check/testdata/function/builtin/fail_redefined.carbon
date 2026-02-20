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
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.8ef: Core.Form = init_form %i32, call_param2 [concrete]
// CHECK:STDOUT:   %A.type.8165c1.1: type = fn_type @A.loc15 [concrete]
// CHECK:STDOUT:   %A.8aef9d.1: %A.type.8165c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.8165c1.2: type = fn_type @A.loc23 [concrete]
// CHECK:STDOUT:   %A.8aef9d.2: %A.type.8165c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %B.type.d6bcfe.1: type = fn_type @B.loc25 [concrete]
// CHECK:STDOUT:   %B.5d732e.1: %B.type.d6bcfe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.d6bcfe.2: type = fn_type @B.loc33 [concrete]
// CHECK:STDOUT:   %B.5d732e.2: %B.type.d6bcfe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.0036b9.1: type = fn_type @C.loc35 [concrete]
// CHECK:STDOUT:   %C.6a91b4.1: %C.type.0036b9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.0036b9.2: type = fn_type @C.loc43 [concrete]
// CHECK:STDOUT:   %C.6a91b4.2: %C.type.0036b9.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %A.decl.loc15: %A.type.8165c1.1 = fn_decl @A.loc15 [concrete = constants.%A.8aef9d.1] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_25: Core.Form = init_form %i32.loc15_25, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc15_9: type = splice_block %i32.loc15_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc15_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc15_17: type = splice_block %i32.loc15_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc15_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc23: %A.type.8165c1.2 = fn_decl @A.loc23 [concrete = constants.%A.8aef9d.2] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc23_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23_32: Core.Form = init_form %i32.loc23_32, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc23_9: type = splice_block %i32.loc23_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc23_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc23_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc23_24: type = splice_block %i32.loc23_24 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc23_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc23_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc25: %B.type.d6bcfe.1 = fn_decl @B.loc25 [concrete = constants.%B.5d732e.1] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc25_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc25_32: Core.Form = init_form %i32.loc25_32, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc25_9: type = splice_block %i32.loc25_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc25_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc25_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc25_24: type = splice_block %i32.loc25_24 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc25_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc25_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc33: %B.type.d6bcfe.2 = fn_decl @B.loc33 [concrete = constants.%B.5d732e.2] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc33_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc33_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc33_25: Core.Form = init_form %i32.loc33_25, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc33_9: type = splice_block %i32.loc33_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc33_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc33_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc33_17: type = splice_block %i32.loc33_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc33_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc33_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc35: %C.type.0036b9.1 = fn_decl @C.loc35 [concrete = constants.%C.6a91b4.1] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc35_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc35_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc35_25: Core.Form = init_form %i32.loc35_25, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc35_9: type = splice_block %i32.loc35_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc35_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc35_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc35_17: type = splice_block %i32.loc35_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc35_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc35_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: %i32 = value_binding m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc43: %C.type.0036b9.2 = fn_decl @C.loc43 [concrete = constants.%C.6a91b4.2] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = value_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.param_patt: %pattern_type.7ce = value_param_pattern %m.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc43_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc43_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc43_25: Core.Form = init_form %i32.loc43_25, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc43_9: type = splice_block %i32.loc43_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc43_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc43_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc43_17: type = splice_block %i32.loc43_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc43_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc43_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
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
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
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
