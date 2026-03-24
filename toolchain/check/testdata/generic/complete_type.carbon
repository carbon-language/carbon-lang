// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/complete_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/complete_type.carbon

// --- fail_incomplete_in_class.carbon

library "[[@TEST_NAME]]";

class B;

class A(T:! type) {
  var v: T;
}

// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+17]]:13: error: parameter has incomplete type `A(B)` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn F(unused x: A(B)) {}
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-6]]:10: note: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
// CHECK:STDERR:   var v: T;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-12]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+7]]:13: error: parameter has incomplete type `A(B)` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn F(unused x: A(B)) {}
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-19]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn F(unused x: A(B)) {}

class B {}

// --- incomplete_in_function.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type, v: T*) {
  *v;
}

// F(B) isn't resolved until the end of the file.
fn G(p: B*) { F(B, p); }

class B {}

// --- fail_incomplete_in_function_at_eof.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type, p: T*) {
  let _: T = *p;
}

// CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+10]]:15: error: unable to monomorphize specific `F(B)` [ResolvingSpecificHere]
// CHECK:STDERR: fn G(p: B*) { F(B, p); }
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE-6]]:10: note: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
// CHECK:STDERR:   let _: T = *p;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE-12]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn G(p: B*) { F(B, p); }

// CHECK:STDOUT: --- fail_incomplete_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.95c: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A.elem.8a2: type = unbound_element_type %A.95c, %T [symbolic]
// CHECK:STDOUT:   %struct_type.v.a4d: type = struct_type {.v: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.136: <witness> = complete_type_witness %struct_type.v.a4d [symbolic]
// CHECK:STDOUT:   %A.764: type = class_type @A, @A(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.fa6: type = pattern_type %A.764 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem.46a: type = unbound_element_type %A.764, %B [concrete]
// CHECK:STDOUT:   %struct_type.v.b74: type = struct_type {.v: %B} [concrete]
// CHECK:STDOUT:   %complete_type.aac: <witness> = complete_type_witness %struct_type.v.b74 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc19
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc19: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_13.1: type = splice_block %.loc21_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.fa6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.fa6 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A.764 = value_param call_param0
// CHECK:STDOUT:     %.loc42: type = splice_block %A [concrete = constants.%A.764] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl.loc19 [concrete = constants.%B]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%B) [concrete = constants.%A.764]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %A.764 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc44: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc21_9.2: type) {
// CHECK:STDOUT:   %T.loc21_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc21_9.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc21_9.1) [symbolic = %A (constants.%A.95c)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %T.loc21_9.1 [symbolic = %A.elem (constants.%A.elem.8a2)]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: @A.%T.loc21_9.1 (%T)} [symbolic = %struct_type.v (constants.%struct_type.v.a4d)]
// CHECK:STDOUT:   %complete_type.loc23_1.2: <witness> = complete_type_witness %struct_type.v [symbolic = %complete_type.loc23_1.2 (constants.%complete_type.136)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_9.2 [symbolic = %T.loc21_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc22: @A.%A.elem (%A.elem.8a2) = field_decl v, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc23_1.1: <witness> = complete_type_witness constants.%struct_type.v.a4d [symbolic = %complete_type.loc23_1.2 (constants.%complete_type.136)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc23_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.95c
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .v = %.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.764) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc21_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%B) {
// CHECK:STDOUT:   %T.loc21_9.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A => constants.%A.764
// CHECK:STDOUT:   %A.elem => constants.%A.elem.46a
// CHECK:STDOUT:   %struct_type.v => constants.%struct_type.v.b74
// CHECK:STDOUT:   %complete_type.loc23_1.2 => constants.%complete_type.aac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.191: type = pattern_type %ptr.27c [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %complete_type.04a: <witness> = complete_type_witness %ptr.27c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc50
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc50: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %v.param_patt: @F.%pattern_type (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:     %v.patt: @F.%pattern_type (%pattern_type.4f4) = at_binding_pattern v, %v.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc52_10.1: type = splice_block %.loc52_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc52_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc52_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc52_6.1 (constants.%T)]
// CHECK:STDOUT:     %v.param: @F.%ptr.loc52_20.1 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:     %.loc52_20: type = splice_block %ptr.loc52_20.2 [symbolic = %ptr.loc52_20.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc52_6.2 [symbolic = %T.loc52_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc52_20.2: type = ptr_type %T.ref [symbolic = %ptr.loc52_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.%ptr.loc52_20.1 (%ptr.e8f) = value_binding v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.191 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.191 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.27c = value_param call_param0
// CHECK:STDOUT:     %.loc57: type = splice_block %ptr [concrete = constants.%ptr.27c] {
// CHECK:STDOUT:       %B.ref.loc57_9: type = name_ref B, file.%B.decl.loc50 [concrete = constants.%B]
// CHECK:STDOUT:       %ptr: type = ptr_type %B.ref.loc57_9 [concrete = constants.%ptr.27c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.27c = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc59: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc52_6.2: type) {
// CHECK:STDOUT:   %T.loc52_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc52_6.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc52_20.1: type = ptr_type %T.loc52_6.1 [symbolic = %ptr.loc52_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc52_20.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc52_20.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%v.param: @F.%ptr.loc52_20.1 (%ptr.e8f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.ref: @F.%ptr.loc52_20.1 (%ptr.e8f) = name_ref v, %v
// CHECK:STDOUT:     %.loc53: ref @F.%T.loc52_6.1 (%T) = deref %v.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.27c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref.loc57_17: type = name_ref B, file.%B.decl.loc50 [concrete = constants.%B]
// CHECK:STDOUT:   %p.ref: %ptr.27c = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc52_6.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc52_20.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc52_6.1 => constants.%B
// CHECK:STDOUT:   %ptr.loc52_20.1 => constants.%ptr.27c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.191
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_in_function_at_eof.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.191: type = pattern_type %ptr.27c [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.27c [concrete]
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc67 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc67 (%pattern_type.4f4) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc67_10.1: type = splice_block %.loc67_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc67_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc67_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc67_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc67_20.1 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:     %.loc67_20: type = splice_block %ptr.loc67_20.2 [symbolic = %ptr.loc67_20.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:       %T.ref.loc67: type = name_ref T, %T.loc67_6.2 [symbolic = %T.loc67_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc67_20.2: type = ptr_type %T.ref.loc67 [symbolic = %ptr.loc67_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc67_20.1 (%ptr.e8f) = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.191 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.191 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.27c = value_param call_param0
// CHECK:STDOUT:     %.loc81: type = splice_block %ptr [concrete = constants.%ptr.27c] {
// CHECK:STDOUT:       %B.ref.loc81_9: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr: type = ptr_type %B.ref.loc81_9 [concrete = constants.%ptr.27c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.27c = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc67_6.2: type) {
// CHECK:STDOUT:   %T.loc67_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc67_6.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc67_20.1: type = ptr_type %T.loc67_6.1 [symbolic = %ptr.loc67_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type.loc67: type = pattern_type %ptr.loc67_20.1 [symbolic = %pattern_type.loc67 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %ptr.loc67_20.1 [symbolic = %require_complete.loc67 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc68: <witness> = require_complete_type %T.loc67_6.1 [symbolic = %require_complete.loc68 (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type.loc68: type = pattern_type %T.loc67_6.1 [symbolic = %pattern_type.loc68 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc67_20.1 (%ptr.e8f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: @F.%pattern_type.loc68 (%pattern_type.51d) = value_binding_pattern _ [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc67_20.1 (%ptr.e8f) = name_ref p, %p
// CHECK:STDOUT:     %.loc68_14.1: ref @F.%T.loc67_6.1 (%T) = deref %p.ref
// CHECK:STDOUT:     %T.ref.loc68: type = name_ref T, %T.loc67_6.2 [symbolic = %T.loc67_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc68_14.2: @F.%T.loc67_6.1 (%T) = acquire_value %.loc68_14.1
// CHECK:STDOUT:     %_: @F.%T.loc67_6.1 (%T) = value_binding _, %.loc68_14.2
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.27c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref.loc81_17: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %p.ref: %ptr.27c = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc67_6.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc67_20.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type.loc67 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc67_6.1 => constants.%B
// CHECK:STDOUT:   %ptr.loc67_20.1 => constants.%ptr.27c
// CHECK:STDOUT:   %pattern_type.loc67 => constants.%pattern_type.191
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc68 => <error>
// CHECK:STDOUT:   %pattern_type.loc68 => constants.%pattern_type.1f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
