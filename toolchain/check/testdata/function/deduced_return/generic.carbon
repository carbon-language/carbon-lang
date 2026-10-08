// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/generic.carbon

// --- generic_definition.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[T: Core.Copy](x: T) -> auto { return x; }
//@dump-sem-ir-end

// --- fail_todo_call_generic.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> auto { return x; }

// CHECK:STDERR: fail_todo_call_generic.carbon:[[@LINE+4]]:24: error: semantics TODO: `call to generic function with deduced return type` [SemanticsTodo]
// CHECK:STDERR: fn G() -> i32 { return F(0 as i32); }
// CHECK:STDERR:                        ^~~~~~~~~~~
// CHECK:STDERR:
fn G() -> i32 { return F(0 as i32); }

// CHECK:STDOUT: --- generic_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_7.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc4_21.1: @F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc4_21.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc4_21.1: @F.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc4_21.1 [symbolic = %x.patt.loc4_21.2 (constants.%x.patt)]
// CHECK:STDOUT:     %return.param_patt.loc4_43.1: @F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc4_43.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc4_43.1: @F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc4_43.1, %.loc4_43.6 [symbolic = %return.patt.loc4_43.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc4_23.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.2 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc4_23.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_23.2: type = converted %T.ref, %T.as_type.loc4_23.2 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc4_23.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4_43.6: type = type_literal constants.%T.as_type [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc4_43.7: Core.Form = init_form %.loc4_43.6 [symbolic = %.loc4_43.4 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %return.param: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc4_7.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc4_7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc4_23.1: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc4_23.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc4_21.2: @F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc4_21.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc4_21.2: @F.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc4_21.2 [symbolic = %x.patt.loc4_21.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc4_23.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %.loc4_43.4: Core.Form = init_form %T.as_type.loc4_23.1 [symbolic = %.loc4_43.4 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc4_43.2: @F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc4_43.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc4_43.2: @F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc4_43.2, %T.as_type.loc4_23.1 [symbolic = %return.patt.loc4_43.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc4_7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc4_43.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc4_7.1 [symbolic = %.loc4_43.5 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc4_43.2: @F.%.loc4_43.5 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc4_43.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc4_43.2: <specific function> = specific_impl_function %impl.elem0.loc4_43.2, @Copy.WithSelf.Op(%T.loc4_7.1) [symbolic = %specific_impl_fn.loc4_43.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc4_23.1 (%T.as_type)) -> out %return.param: @F.%T.as_type.loc4_23.1 (%T.as_type) [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc4_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_44: init @F.%T.as_type.loc4_23.1 (%T.as_type) = splice_block %Copy.WithSelf.Op.call {
// CHECK:STDOUT:       %impl.elem0.loc4_43.1: @F.%.loc4_43.5 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc4_43.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:       %bound_method.loc4_43.1: <bound method> = bound_method %x.ref, %impl.elem0.loc4_43.1
// CHECK:STDOUT:       %.loc4_43.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %.loc4_43.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %specific_impl_fn.loc4_43.1: <specific function> = specific_impl_function %impl.elem0.loc4_43.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc4_43.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:       %bound_method.loc4_43.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc4_43.1
// CHECK:STDOUT:       %.loc4_43.3: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:       %Copy.WithSelf.Op.call: init @F.%T.as_type.loc4_23.1 (%T.as_type) to %.loc4_43.3 = call %bound_method.loc4_43.2(%x.ref)
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return %.loc4_44 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc4_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc4_21.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc4_21.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
