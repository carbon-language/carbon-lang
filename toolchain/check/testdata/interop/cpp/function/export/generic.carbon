// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/generic.carbon
// CHECK:STDOUT: --- generic_type_impls_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %self.param_patt.a33: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.696: %pattern_type.c60 = at_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.26f: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.11f: %I.WithSelf.Doit.type.26f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.628: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Doit.decl [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %I.impl_witness.6ad: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit.type: type = fn_type @A.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit: %A.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.64b: %I.type = facet_value %A, (%I.impl_witness.6ad) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.5b3: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.64b) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.dab: %I.WithSelf.Doit.type.5b3 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %I.impl_witness.5b8: <witness> = impl_witness @B.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt.f1f: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4eb: %pattern_type.e39 = at_binding_pattern self, %self.param_patt.f1f [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit.type: type = fn_type @B.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit: %B.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.9f8: %I.type = facet_value %B, (%I.impl_witness.5b8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.2ce: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.9f8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.450: %I.WithSelf.Doit.type.2ce = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.b3a = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.860: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.45d: %I.WithSelf.Doit.type.860 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.8a9: type = fn_type_with_self_type %I.WithSelf.Doit.type.860, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.8a9 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.300: %pattern_type.b3a = at_binding_pattern self, %self.param_patt.58e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Doit(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc251_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e5: %Destroy.type = facet_value %A, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.09e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6e5) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.09e, %Destroy.facet.6e5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc256 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f70: %Destroy.type = facet_value %B, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cda: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f70) [concrete]
// CHECK:STDOUT:   %.3d9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cda, %Destroy.facet.f70 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc266_7.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc266_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc266_14.1: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc266_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc266_14.1: @F.%pattern_type (%pattern_type.b3a) = at_binding_pattern t, %t.param_patt.loc266_14.1 [symbolic = %t.patt.loc266_14.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc266_10: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc266_7.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc266_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc266_16.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc266_16.1: type = splice_block %.loc266_16.2 [symbolic = %T.as_type.loc266_16.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc266_7.2 [symbolic = %T.loc266_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc266_16.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc266_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc266_16.2: type = converted %T.ref, %T.as_type.loc266_16.2 [symbolic = %T.as_type.loc266_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc266_16.1 (%T.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]

// --- generic_type_impls_interface.carbon
// CHECK:STDOUT:   inline_cpp "void G() {\n  Carbon::A a;\n  Carbon::B b;\n  // CHECK:STDERR: generic_type_impls_interface.carbon:{{\[\[}}@LINE+4]]:11: error: no type named 'F' in namespace 'Carbon' [CppInteropParseError]\n  // CHECK:STDERR:   278 |   Carbon::F(a);\n  // CHECK:STDERR:       |   ~~~~~~~~^\n  // CHECK:STDERR:\n  Carbon::F(a);\n  // CHECK:STDERR: generic_type_impls_interface.carbon:{{\[\[}}@LINE+4]]:11: error: no type named 'F' in namespace 'Carbon' [CppInteropParseError]\n  // CHECK:STDERR:   283 |   Carbon::F(b);\n  // CHECK:STDERR:       |   ~~~~~~~~^\n  // CHECK:STDERR:\n  Carbon::F(b);\n}\n"
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.Doit.decl: @I.WithSelf.%I.WithSelf.Doit.type (%I.WithSelf.Doit.type.26f) = fn_decl @I.WithSelf.Doit [symbolic = @I.WithSelf.%I.WithSelf.Doit (constants.%I.WithSelf.Doit.11f)] {
// CHECK:STDOUT:     %self.param_patt.loc248_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc248_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:     %self.patt.loc248_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc248_11.1 [symbolic = %self.patt.loc248_11.2 (constants.%self.patt.696)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.Doit.%Self.as_type.loc248_11.1 (%Self.as_type.c84) = value_param call_param0
// CHECK:STDOUT:     %.loc248_11.1: type = splice_block %.loc248_11.2 [symbolic = %Self.as_type.loc248_11.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc248_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc248_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc248_11.2: type = converted %Self.ref, %Self.as_type.loc248_11.2 [symbolic = %Self.as_type.loc248_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.Doit.%Self.as_type.loc248_11.1 (%Self.as_type.c84) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.Doit.decl [concrete = constants.%assoc0.628]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Doit = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.Doit.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %A.as.I.impl.Doit.decl: %A.as.I.impl.Doit.type = fn_decl @A.as.I.impl.Doit [concrete = constants.%A.as.I.impl.Doit] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%A.as.I.impl.Doit.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.6ad]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Doit = %A.as.I.impl.Doit.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %B.as.I.impl.Doit.decl: %B.as.I.impl.Doit.type = fn_decl @B.as.I.impl.Doit [concrete = constants.%B.as.I.impl.Doit] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete = constants.%self.param_patt.f1f]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4eb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%B.as.I.impl.Doit.decl), @B.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.5b8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Doit = %B.as.I.impl.Doit.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Doit(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %Self.as_type.loc248_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc248_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc248_11.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:   %self.param_patt.loc248_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc248_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:   %self.patt.loc248_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc248_11.2 [symbolic = %self.patt.loc248_11.2 (constants.%self.patt.696)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.Doit.%Self.as_type.loc248_11.1 (%Self.as_type.c84));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.I.impl.Doit(%self.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.I.impl.Doit(%self.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc266_7.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc266_7.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc266_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc266_7.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc266_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc266_16.1: type = facet_access_type %T.loc266_7.1 [symbolic = %T.as_type.loc266_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc266_16.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %t.param_patt.loc266_14.2: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc266_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc266_14.2: @F.%pattern_type (%pattern_type.b3a) = at_binding_pattern t, %t.param_patt.loc266_14.2 [symbolic = %t.patt.loc266_14.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc266_16.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.WithSelf.Doit.type: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T.loc266_7.1) [symbolic = %I.WithSelf.Doit.type (constants.%I.WithSelf.Doit.type.860)]
// CHECK:STDOUT:   %.loc267_4: type = fn_type_with_self_type %I.WithSelf.Doit.type, %T.loc266_7.1 [symbolic = %.loc267_4 (constants.%.8a9)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc266_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc267_4.2: @F.%.loc267_4 (%.8a9) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc267_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc267_4.2: <specific function> = specific_impl_function %impl.elem0.loc267_4.2, @I.WithSelf.Doit(%T.loc266_7.1) [symbolic = %specific_impl_fn.loc267_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc266_16.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc266_16.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Doit.ref: %I.assoc_type = name_ref Doit, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.628]
// CHECK:STDOUT:     %impl.elem0.loc267_4.1: @F.%.loc267_4 (%.8a9) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc267_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc267_4: <bound method> = bound_method %t.ref, %impl.elem0.loc267_4.1
// CHECK:STDOUT:     %.loc267_10.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc266_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc267_10.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc266_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc267_4.1: <specific function> = specific_impl_function %impl.elem0.loc267_4.1, @I.WithSelf.Doit(constants.%T) [symbolic = %specific_impl_fn.loc267_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc267_10: <bound method> = bound_method %t.ref, %specific_impl_fn.loc267_4.1
// CHECK:STDOUT:     %I.WithSelf.Doit.call: init %empty_tuple.type = call %bound_method.loc267_10(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.__destroy_thunk(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.dca = impl_witness_access constants.%custom_witness.df9cc1.2, element0 [concrete = constants.%Destroy.Op.1a2547.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc251_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc251_9.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.__destroy_thunk(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.3d9 = impl_witness_access constants.%custom_witness.df9cc1.3, element0 [concrete = constants.%Destroy.Op.1a2547.3]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc256(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.26f
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.11f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%Self.37e) {
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type.loc248_11.1 => constants.%Self.as_type.c84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c60
// CHECK:STDOUT:   %self.param_patt.loc248_11.2 => constants.%self.param_patt.a33
// CHECK:STDOUT:   %self.patt.loc248_11.2 => constants.%self.patt.696
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.64b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.64b
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.5b3
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.dab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.64b) {
// CHECK:STDOUT:   %Self => constants.%I.facet.64b
// CHECK:STDOUT:   %Self.as_type.loc248_11.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc248_11.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc248_11.2 => constants.%self.patt.312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.9f8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.9f8
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.2ce
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.450
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.9f8) {
// CHECK:STDOUT:   %Self => constants.%I.facet.9f8
// CHECK:STDOUT:   %Self.as_type.loc248_11.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %self.param_patt.loc248_11.2 => constants.%self.param_patt.f1f
// CHECK:STDOUT:   %self.patt.loc248_11.2 => constants.%self.patt.4eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc266_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc266_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc266_16.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %t.param_patt.loc266_14.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc266_14.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.860
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.45d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc248_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %self.param_patt.loc248_11.2 => constants.%self.param_patt.58e
// CHECK:STDOUT:   %self.patt.loc248_11.2 => constants.%self.patt.300
// CHECK:STDOUT: }
// CHECK:STDOUT:
library "[[@TEST_NAME]]";
import Cpp;

//@dump-sem-ir-begin
interface I {
  fn Doit(self);
}

class A {
  impl as I {
    fn Doit(unused self) {}
  }
}
class B {
  impl as I {
    fn Doit(unused self) {}
  }
}

fn F[T:! I](t: T) {
  t.Doit();
}

inline Cpp '''
void G() {
  Carbon::A a;
  Carbon::B b;
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE+7]]:3: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:   278 |   Carbon::F(a);
  // CHECK:STDERR:       |   ^~~~~~~~~
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE-11]]:19: note: candidate template ignored: substitution failure [with T = Carbon::A] [CppInteropParseNote]
  // CHECK:STDERR:   266 | fn F[T:! I](t: T) {
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR:
  Carbon::F(a);
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE+7]]:3: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:   283 |   Carbon::F(b);
  // CHECK:STDERR:       |   ^~~~~~~~~
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE-19]]:19: note: candidate template ignored: substitution failure [with T = Carbon::B] [CppInteropParseNote]
  // CHECK:STDERR:   266 | fn F[T:! I](t: T) {
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR:
  Carbon::F(b);
}
''';
//@dump-sem-ir-end

