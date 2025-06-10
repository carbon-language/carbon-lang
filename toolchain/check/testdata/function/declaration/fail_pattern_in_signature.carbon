// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --no-prelude-import --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_pattern_in_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_pattern_in_signature.carbon

// CHECK:STDERR: fail_pattern_in_signature.carbon:[[@LINE+4]]:6: error: semantics TODO: `Support patterns besides bindings in parameter list` [SemanticsTodo]
// CHECK:STDERR: fn F((a: {}, b: {}), c: {});
// CHECK:STDERR:      ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F((a: {}, b: {}), c: {});

// CHECK:STDOUT: --- fail_pattern_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.a96 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.a96 = value_param_pattern %a.patt, call_param<none> [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.a96 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.a96 = value_param_pattern %b.patt, call_param<none> [concrete]
// CHECK:STDOUT:     %.loc18_19: %pattern_type.de4 = tuple_pattern (%a.param_patt, %b.param_patt) [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.a96 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.a96 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc18_26.1: type = splice_block %.loc18_26.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc18_26.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc18_26.3: type = converted %.loc18_26.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %empty_struct_type = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %empty_struct_type);
// CHECK:STDOUT:
