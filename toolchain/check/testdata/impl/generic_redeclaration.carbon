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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/generic_redeclaration.carbon

// --- fail_todo_same_self_and_interface.carbon

library "[[@TEST_NAME]]";

interface Interface {}

interface I {}
interface J {}
interface K {}
interface L {}

// TODO: Put these in a match_first so the test will pass again.
impl forall [T: I] T as Interface;
impl forall [T: J] T as Interface;
impl forall [T: K] T as Interface;
impl forall [T: L] T as Interface;

// These are different impls, so they are not redefinitions, even though the
// self type and constraint type are the same.
impl forall [T: I] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+3]]:1-35: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-2]]:1-35: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:
impl forall [T: J] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+3]]:1-35: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-6]]:1-35: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:
impl forall [T: K] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+3]]:1-35: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-10]]:1-35: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:
impl forall [T: L] T as Interface {}

// --- fail_same_self_and_interface_redefined.carbon

library "[[@TEST_NAME]]";

interface I {}
interface J {}

impl forall [T: I] T as J {}
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+3]]:1-27: error: redefinition of `impl T as J` [ImplRedefinition]
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE-2]]:1-27: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR:
impl forall [T: I] T as J {}

// --- fail_same_type_different_spelling.carbon

library "[[@TEST_NAME]]";

class C;
interface I {}

// Since the spelling is different, it's not caught as a redefinition, but it
// is diagnosed as an overlapping impl without using a `match_first` block.
impl C as I {}
// CHECK:STDERR: fail_same_type_different_spelling.carbon:[[@LINE+3]]:1-20: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: fail_same_type_different_spelling.carbon:[[@LINE-2]]:1-13: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:
impl (C, C).0 as I {}

// --- fail_redefinition_generic_regions.carbon

interface I {}

impl forall [T: type] T as I {
  fn A() {}
}

// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE+3]]:1-30: error: redefinition of `impl T as I` [ImplRedefinition]
// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE-5]]:1-30: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR:
impl forall [T: type] T as I {
  // Although not referenced, B has the same generic region index of A, and
  // makes C a different index. We used to merge, and this was a crash.
  fn B() {}
  fn C() -> () { return (); }
  fn D() -> () {
    return C();
  }
}

// CHECK:STDOUT: --- fail_todo_same_self_and_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.3ce: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.a32: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.480: %L.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.b38: type = facet_access_type %T.dda [symbolic]
// CHECK:STDOUT:   %.915: <witness> = impl_self_witness %T.as_type.b38, @Interface [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.740: <witness> = impl_witness @T.as_type.as.Interface.impl.72f.%Interface.impl_witness_table, @T.as_type.as.Interface.impl.72f(%T.dda) [symbolic]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt.ac1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e40: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.d90: type = facet_access_type %T.e40 [symbolic]
// CHECK:STDOUT:   %.eab: <witness> = impl_self_witness %T.as_type.d90, @Interface [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.cb9: <witness> = impl_witness @T.as_type.as.Interface.impl.24e.%Interface.impl_witness_table, @T.as_type.as.Interface.impl.24e(%T.e40) [symbolic]
// CHECK:STDOUT:   %pattern_type.b64: type = pattern_type %K.type [concrete]
// CHECK:STDOUT:   %T.patt.0ef: %pattern_type.b64 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.062: %K.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.1ea: type = facet_access_type %T.062 [symbolic]
// CHECK:STDOUT:   %.2ae: <witness> = impl_self_witness %T.as_type.1ea, @Interface [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.e2e: <witness> = impl_witness @T.as_type.as.Interface.impl.b21.%Interface.impl_witness_table, @T.as_type.as.Interface.impl.b21(%T.062) [symbolic]
// CHECK:STDOUT:   %pattern_type.25e: type = pattern_type %L.type [concrete]
// CHECK:STDOUT:   %T.patt.b11: %pattern_type.25e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.709: %L.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.56f: type = facet_access_type %T.709 [symbolic]
// CHECK:STDOUT:   %.30d: <witness> = impl_self_witness %T.as_type.56f, @Interface [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.3f0: <witness> = impl_witness @T.as_type.as.Interface.impl.2b4.%Interface.impl_witness_table, @T.as_type.as.Interface.impl.2b4(%T.709) [symbolic]
// CHECK:STDOUT:   %Interface.facet.6a0: %Interface.type = facet_value %T.as_type.b38, (%Interface.impl_witness.740) [symbolic]
// CHECK:STDOUT:   %Interface.facet.316: %Interface.type = facet_value %T.as_type.d90, (%Interface.impl_witness.cb9) [symbolic]
// CHECK:STDOUT:   %Interface.facet.a30: %Interface.type = facet_value %T.as_type.1ea, (%Interface.impl_witness.e2e) [symbolic]
// CHECK:STDOUT:   %Interface.facet.edc: %Interface.type = facet_value %T.as_type.56f, (%Interface.impl_witness.3f0) [symbolic]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.72f [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12: %I.type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T.dda)]
// CHECK:STDOUT:     %T.as_type.loc12_20.1: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc12_20.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %.loc12_20: type = converted %T.ref.loc12, %T.as_type.loc12_20.1 [symbolic = %T.as_type.loc12_20.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %Interface.ref.loc12: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc12_17: type = splice_block %I.ref.loc12 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen.loc12: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref.loc12: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.dda)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @T.as_type.as.Interface.impl.72f.%.loc12_20, @Interface [symbolic = @T.as_type.as.Interface.impl.72f.%.loc12_34 (constants.%.915)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.24e [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13: %J.type = name_ref T, %T.loc13_15.1 [symbolic = %T.loc13_15.2 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc13_20.1: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_20.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %.loc13_20: type = converted %T.ref.loc13, %T.as_type.loc13_20.1 [symbolic = %T.as_type.loc13_20.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %Interface.ref.loc13: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc13_17: type = splice_block %J.ref.loc13 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen.loc13: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.2 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @T.as_type.as.Interface.impl.24e.%.loc13_20, @Interface [symbolic = @T.as_type.as.Interface.impl.24e.%.loc13_34 (constants.%.eab)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.b21 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type.b64 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.0ef)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc14: %K.type = name_ref T, %T.loc14_15.1 [symbolic = %T.loc14_15.2 (constants.%T.062)]
// CHECK:STDOUT:     %T.as_type.loc14_20.1: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_20.2 (constants.%T.as_type.1ea)]
// CHECK:STDOUT:     %.loc14_20: type = converted %T.ref.loc14, %T.as_type.loc14_20.1 [symbolic = %T.as_type.loc14_20.2 (constants.%T.as_type.1ea)]
// CHECK:STDOUT:     %Interface.ref.loc14: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc14_17: type = splice_block %K.ref.loc14 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.frozen.loc14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %K.ref.loc14: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.1: %K.type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T.062)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @T.as_type.as.Interface.impl.b21.%.loc14_20, @Interface [symbolic = @T.as_type.as.Interface.impl.b21.%.loc14_34 (constants.%.2ae)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.2b4 [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_15.1: %pattern_type.25e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_15.2 (constants.%T.patt.b11)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc15: %L.type = name_ref T, %T.loc15_15.1 [symbolic = %T.loc15_15.2 (constants.%T.709)]
// CHECK:STDOUT:     %T.as_type.loc15_20.1: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc15_20.2 (constants.%T.as_type.56f)]
// CHECK:STDOUT:     %.loc15_20: type = converted %T.ref.loc15, %T.as_type.loc15_20.1 [symbolic = %T.as_type.loc15_20.2 (constants.%T.as_type.56f)]
// CHECK:STDOUT:     %Interface.ref.loc15: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc15_17: type = splice_block %L.ref.loc15 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.frozen.loc15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %L.ref.loc15: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_15.1: %L.type = symbolic_binding T, 0 [symbolic = %T.loc15_15.2 (constants.%T.709)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @T.as_type.as.Interface.impl.2b4.%.loc15_20, @Interface [symbolic = @T.as_type.as.Interface.impl.2b4.%.loc15_34 (constants.%.30d)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.72f [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: %I.type = name_ref T, %T.loc19 [symbolic = %T.loc12_15.2 (constants.%T.dda)]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc12_20.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %.loc19_20: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc12_20.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %Interface.ref.loc19: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc19_17: type = splice_block %I.ref.loc19 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen.loc19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref.loc19: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19: %I.type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.dda)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = impl_self_witness @T.as_type.as.Interface.impl.72f.%.loc19_20, @Interface [symbolic = @T.as_type.as.Interface.impl.72f.%.loc12_34 (constants.%.915)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.24e [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc23: %J.type = name_ref T, %T.loc23 [symbolic = %T.loc13_15.2 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc23: type = facet_access_type %T.ref.loc23 [symbolic = %T.as_type.loc13_20.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %.loc23_20: type = converted %T.ref.loc23, %T.as_type.loc23 [symbolic = %T.as_type.loc13_20.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %Interface.ref.loc23: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc23_17: type = splice_block %J.ref.loc23 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen.loc23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref.loc23: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc23: %J.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.2 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = impl_self_witness @T.as_type.as.Interface.impl.24e.%.loc23_20, @Interface [symbolic = @T.as_type.as.Interface.impl.24e.%.loc13_34 (constants.%.eab)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.b21 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type.b64 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.0ef)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc27: %K.type = name_ref T, %T.loc27 [symbolic = %T.loc14_15.2 (constants.%T.062)]
// CHECK:STDOUT:     %T.as_type.loc27: type = facet_access_type %T.ref.loc27 [symbolic = %T.as_type.loc14_20.2 (constants.%T.as_type.1ea)]
// CHECK:STDOUT:     %.loc27_20: type = converted %T.ref.loc27, %T.as_type.loc27 [symbolic = %T.as_type.loc14_20.2 (constants.%T.as_type.1ea)]
// CHECK:STDOUT:     %Interface.ref.loc27: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc27_17: type = splice_block %K.ref.loc27 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.frozen.loc27: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %K.ref.loc27: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27: %K.type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T.062)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27: <witness> = impl_self_witness @T.as_type.as.Interface.impl.b21.%.loc27_20, @Interface [symbolic = @T.as_type.as.Interface.impl.b21.%.loc14_34 (constants.%.2ae)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.Interface.impl.2b4 [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_15.1: %pattern_type.25e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_15.2 (constants.%T.patt.b11)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc31: %L.type = name_ref T, %T.loc31 [symbolic = %T.loc15_15.2 (constants.%T.709)]
// CHECK:STDOUT:     %T.as_type.loc31: type = facet_access_type %T.ref.loc31 [symbolic = %T.as_type.loc15_20.2 (constants.%T.as_type.56f)]
// CHECK:STDOUT:     %.loc31_20: type = converted %T.ref.loc31, %T.as_type.loc31 [symbolic = %T.as_type.loc15_20.2 (constants.%T.as_type.56f)]
// CHECK:STDOUT:     %Interface.ref.loc31: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc31_17: type = splice_block %L.ref.loc31 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.frozen.loc31: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %L.ref.loc31: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31: %L.type = symbolic_binding T, 0 [symbolic = %T.loc15_15.2 (constants.%T.709)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31: <witness> = impl_self_witness @T.as_type.as.Interface.impl.2b4.%.loc31_20, @Interface [symbolic = @T.as_type.as.Interface.impl.2b4.%.loc15_34 (constants.%.30d)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a32]
// CHECK:STDOUT:   %K.WithSelf.decl = interface_with_self_decl @K [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = symbolic_binding Self, 0 [symbolic = constants.%Self.480]
// CHECK:STDOUT:   %L.WithSelf.decl = interface_with_self_decl @L [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Interface.impl.72f(%T.loc12_15.1: %I.type) {
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc12_15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.dda)]
// CHECK:STDOUT:   %T.as_type.loc12_20.2: type = facet_access_type %T.loc12_15.2 [symbolic = %T.as_type.loc12_20.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:   %.loc12_34: <witness> = impl_self_witness %T.as_type.loc12_20.2, @Interface [symbolic = %.loc12_34 (constants.%.915)]
// CHECK:STDOUT:   %Interface.impl_witness.loc12_34.2: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.72f(%T.loc12_15.2) [symbolic = %Interface.impl_witness.loc12_34.2 (constants.%Interface.impl_witness.740)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12_20 as %Interface.ref.loc12 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.as_type.as.Interface.impl.72f [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc12_34.1: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.72f(constants.%T.dda) [symbolic = %Interface.impl_witness.loc12_34.2 (constants.%Interface.impl_witness.740)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Interface.ref.loc12
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc12_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Interface.impl.24e(%T.loc13_15.1: %J.type) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc13_15.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.2 (constants.%T.e40)]
// CHECK:STDOUT:   %T.as_type.loc13_20.2: type = facet_access_type %T.loc13_15.2 [symbolic = %T.as_type.loc13_20.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %.loc13_34: <witness> = impl_self_witness %T.as_type.loc13_20.2, @Interface [symbolic = %.loc13_34 (constants.%.eab)]
// CHECK:STDOUT:   %Interface.impl_witness.loc13_34.2: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.24e(%T.loc13_15.2) [symbolic = %Interface.impl_witness.loc13_34.2 (constants.%Interface.impl_witness.cb9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13_20 as %Interface.ref.loc13 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.as_type.as.Interface.impl.24e [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc13_34.1: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.24e(constants.%T.e40) [symbolic = %Interface.impl_witness.loc13_34.2 (constants.%Interface.impl_witness.cb9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Interface.ref.loc13
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc13_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Interface.impl.b21(%T.loc14_15.1: %K.type) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type.b64 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.0ef)]
// CHECK:STDOUT:   %T.loc14_15.2: %K.type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T.062)]
// CHECK:STDOUT:   %T.as_type.loc14_20.2: type = facet_access_type %T.loc14_15.2 [symbolic = %T.as_type.loc14_20.2 (constants.%T.as_type.1ea)]
// CHECK:STDOUT:   %.loc14_34: <witness> = impl_self_witness %T.as_type.loc14_20.2, @Interface [symbolic = %.loc14_34 (constants.%.2ae)]
// CHECK:STDOUT:   %Interface.impl_witness.loc14_34.2: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.b21(%T.loc14_15.2) [symbolic = %Interface.impl_witness.loc14_34.2 (constants.%Interface.impl_witness.e2e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_20 as %Interface.ref.loc14 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.as_type.as.Interface.impl.b21 [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc14_34.1: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.b21(constants.%T.062) [symbolic = %Interface.impl_witness.loc14_34.2 (constants.%Interface.impl_witness.e2e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Interface.ref.loc14
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc14_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Interface.impl.2b4(%T.loc15_15.1: %L.type) {
// CHECK:STDOUT:   %T.patt.loc15_15.2: %pattern_type.25e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_15.2 (constants.%T.patt.b11)]
// CHECK:STDOUT:   %T.loc15_15.2: %L.type = symbolic_binding T, 0 [symbolic = %T.loc15_15.2 (constants.%T.709)]
// CHECK:STDOUT:   %T.as_type.loc15_20.2: type = facet_access_type %T.loc15_15.2 [symbolic = %T.as_type.loc15_20.2 (constants.%T.as_type.56f)]
// CHECK:STDOUT:   %.loc15_34: <witness> = impl_self_witness %T.as_type.loc15_20.2, @Interface [symbolic = %.loc15_34 (constants.%.30d)]
// CHECK:STDOUT:   %Interface.impl_witness.loc15_34.2: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.2b4(%T.loc15_15.2) [symbolic = %Interface.impl_witness.loc15_34.2 (constants.%Interface.impl_witness.3f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc15_20 as %Interface.ref.loc15 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.as_type.as.Interface.impl.2b4 [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc15_34.1: <witness> = impl_witness %Interface.impl_witness_table, @T.as_type.as.Interface.impl.2b4(constants.%T.709) [symbolic = %Interface.impl_witness.loc15_34.2 (constants.%Interface.impl_witness.3f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Interface.ref.loc15
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc15_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self.3ce) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf(constants.%Self.a32) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%Self.480) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Interface.impl.72f(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc12_15.2 => constants.%T.dda
// CHECK:STDOUT:   %T.as_type.loc12_20.2 => constants.%T.as_type.b38
// CHECK:STDOUT:   %.loc12_34 => constants.%.915
// CHECK:STDOUT:   %Interface.impl_witness.loc12_34.2 => constants.%Interface.impl_witness.740
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Interface.impl.24e(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc13_15.2 => constants.%T.e40
// CHECK:STDOUT:   %T.as_type.loc13_20.2 => constants.%T.as_type.d90
// CHECK:STDOUT:   %.loc13_34 => constants.%.eab
// CHECK:STDOUT:   %Interface.impl_witness.loc13_34.2 => constants.%Interface.impl_witness.cb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Interface.impl.b21(constants.%T.062) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt.0ef
// CHECK:STDOUT:   %T.loc14_15.2 => constants.%T.062
// CHECK:STDOUT:   %T.as_type.loc14_20.2 => constants.%T.as_type.1ea
// CHECK:STDOUT:   %.loc14_34 => constants.%.2ae
// CHECK:STDOUT:   %Interface.impl_witness.loc14_34.2 => constants.%Interface.impl_witness.e2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Interface.impl.2b4(constants.%T.709) {
// CHECK:STDOUT:   %T.patt.loc15_15.2 => constants.%T.patt.b11
// CHECK:STDOUT:   %T.loc15_15.2 => constants.%T.709
// CHECK:STDOUT:   %T.as_type.loc15_20.2 => constants.%T.as_type.56f
// CHECK:STDOUT:   %.loc15_34 => constants.%.30d
// CHECK:STDOUT:   %Interface.impl_witness.loc15_34.2 => constants.%Interface.impl_witness.3f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.6a0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.316) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.a30) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.edc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.989: <witness> = impl_self_witness %T.as_type, @J [symbolic]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @T.as_type.as.J.impl.377648.1.%J.impl_witness_table, @T.as_type.as.J.impl.377648.1(%T) [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type, (%J.impl_witness) [symbolic]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.J.impl.377648.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc7_20.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc7_20: type = converted %T.ref, %T.as_type.loc7_20.1 [symbolic = %T.as_type.loc7_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc7_17: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as_type.as.J.impl.377648.1.%.loc7_20, @J [symbolic = @T.as_type.as.J.impl.377648.1.%.loc7_27 (constants.%.989)]
// CHECK:STDOUT:   impl_decl @T.as_type.as.J.impl.377648.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11_15.1 [symbolic = %T.loc11_15.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc11_20.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_20: type = converted %T.ref, %T.as_type.loc11_20.1 [symbolic = %T.as_type.loc11_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc11_17: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @T.as_type.as.J.impl.377648.2.%.loc11_20, @J [symbolic = @T.as_type.as.J.impl.377648.2.%.loc11_27 (constants.%.989)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.J.impl.377648.1(%T.loc7_15.1: %I.type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_20.2: type = facet_access_type %T.loc7_15.2 [symbolic = %T.as_type.loc7_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc7_27: <witness> = impl_self_witness %T.as_type.loc7_20.2, @J [symbolic = %.loc7_27 (constants.%.989)]
// CHECK:STDOUT:   %J.impl_witness.loc7_27.2: <witness> = impl_witness %J.impl_witness_table, @T.as_type.as.J.impl.377648.1(%T.loc7_15.2) [symbolic = %J.impl_witness.loc7_27.2 (constants.%J.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7_20 as %J.ref {
// CHECK:STDOUT:     %J.impl_witness_table = impl_witness_table (), @T.as_type.as.J.impl.377648.1 [concrete]
// CHECK:STDOUT:     %J.impl_witness.loc7_27.1: <witness> = impl_witness %J.impl_witness_table, @T.as_type.as.J.impl.377648.1(constants.%T) [symbolic = %J.impl_witness.loc7_27.2 (constants.%J.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.ref
// CHECK:STDOUT:     witness = %J.impl_witness.loc7_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.J.impl.377648.2(%T.loc11_15.1: %I.type) {
// CHECK:STDOUT:   %T.patt.loc11_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc11_15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc11_20.2: type = facet_access_type %T.loc11_15.2 [symbolic = %T.as_type.loc11_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc11_27: <witness> = impl_self_witness %T.as_type.loc11_20.2, @J [symbolic = %.loc11_27 (constants.%.989)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_20 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.ref
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.J.impl.377648.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_27 => constants.%.989
// CHECK:STDOUT:   %J.impl_witness.loc7_27.2 => constants.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.J.impl.377648.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc11_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_15.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc11_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %.loc11_27 => constants.%.989
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.df5481.1: <witness> = impl_witness @C.as.I.impl.e103db.1.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.a71ed7.1: %I.type = facet_value %C, (%I.impl_witness.df5481.1) [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I.impl_witness.df5481.2: <witness> = impl_witness @C.as.I.impl.e103db.2.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.a71ed7.2: %I.type = facet_value %C, (%I.impl_witness.df5481.2) [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl.e103db.1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.I.impl.e103db.1.%C.ref, @I [concrete = constants.%.b7f]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.e103db.2 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc13_7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc13_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc13_11.1: %tuple.type = tuple_literal (%C.ref.loc13_7, %C.ref.loc13_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %tuple: %tuple.type = tuple_value (%C.ref.loc13_7, %C.ref.loc13_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_11.2: %tuple.type = converted %.loc13_11.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %tuple.elem0: type = tuple_access %.loc13_11.2, element0 [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.I.impl.e103db.2.%tuple.elem0, @I [concrete = constants.%.b7f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.e103db.1: %C.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.e103db.1 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.df5481.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.e103db.2: %tuple.elem0 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.e103db.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.df5481.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a71ed7.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a71ed7.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefinition_generic_regions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %.917: <witness> = impl_self_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as.I.impl.812d8b.1.%I.impl_witness_table, @T.as.I.impl.812d8b.1(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.812d8b.1(%T) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as.I.impl.A: %T.as.I.impl.A.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.812d8b.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B: %T.as.I.impl.B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.812d8b.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.C: %T.as.I.impl.C.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.812d8b.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D: %T.as.I.impl.D.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.C.specific_fn: <specific function> = specific_function %T.as.I.impl.C, @T.as.I.impl.C(%T) [symbolic]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.I.impl.812d8b.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4: <witness> = impl_self_witness @T.as.I.impl.812d8b.1.%T.ref, @I [symbolic = @T.as.I.impl.812d8b.1.%.loc4_30 (constants.%.917)]
// CHECK:STDOUT:   impl_decl @T.as.I.impl.812d8b.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_15.2 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc11_17.1: type = splice_block %.loc11_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc11_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @T.as.I.impl.812d8b.2.%T.ref, @I [symbolic = @T.as.I.impl.812d8b.2.%.loc11_30 (constants.%.917)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.812d8b.1(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   %.loc4_30: <witness> = impl_self_witness %T.loc4_15.1, @I [symbolic = %.loc4_30 (constants.%.917)]
// CHECK:STDOUT:   %I.impl_witness.loc4_30.2: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl.812d8b.1(%T.loc4_15.1) [symbolic = %I.impl_witness.loc4_30.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.812d8b.1(%T.loc4_15.1) [symbolic = %T.as.I.impl.A.type (constants.%T.as.I.impl.A.type)]
// CHECK:STDOUT:   %T.as.I.impl.A: @T.as.I.impl.812d8b.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = struct_value () [symbolic = %T.as.I.impl.A (constants.%T.as.I.impl.A)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.A.decl: @T.as.I.impl.812d8b.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = fn_decl @T.as.I.impl.A [symbolic = @T.as.I.impl.812d8b.1.%T.as.I.impl.A (constants.%T.as.I.impl.A)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @T.as.I.impl.812d8b.1 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc4_30.1: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl.812d8b.1(constants.%T) [symbolic = %I.impl_witness.loc4_30.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .A = %T.as.I.impl.A.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc4_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.812d8b.2(%T.loc11_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc11_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc11_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:   %.loc11_30: <witness> = impl_self_witness %T.loc11_15.1, @I [symbolic = %.loc11_30 (constants.%.917)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.812d8b.2(%T.loc11_15.1) [symbolic = %T.as.I.impl.B.type (constants.%T.as.I.impl.B.type)]
// CHECK:STDOUT:   %T.as.I.impl.B: @T.as.I.impl.812d8b.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = struct_value () [symbolic = %T.as.I.impl.B (constants.%T.as.I.impl.B)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.812d8b.2(%T.loc11_15.1) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.812d8b.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.812d8b.2(%T.loc11_15.1) [symbolic = %T.as.I.impl.D.type (constants.%T.as.I.impl.D.type)]
// CHECK:STDOUT:   %T.as.I.impl.D: @T.as.I.impl.812d8b.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = struct_value () [symbolic = %T.as.I.impl.D (constants.%T.as.I.impl.D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.B.decl: @T.as.I.impl.812d8b.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = fn_decl @T.as.I.impl.B [symbolic = @T.as.I.impl.812d8b.2.%T.as.I.impl.B (constants.%T.as.I.impl.B)] {} {}
// CHECK:STDOUT:     %T.as.I.impl.C.decl: @T.as.I.impl.812d8b.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = fn_decl @T.as.I.impl.C [symbolic = @T.as.I.impl.812d8b.2.%T.as.I.impl.C (constants.%T.as.I.impl.C)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc15_14.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc15_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc15_14.2: type = converted %.loc15_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc15_14.3: Core.Form = init_form %.loc15_14.2 [concrete = constants.%.262]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.as.I.impl.D.decl: @T.as.I.impl.812d8b.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = fn_decl @T.as.I.impl.D [symbolic = @T.as.I.impl.812d8b.2.%T.as.I.impl.D (constants.%T.as.I.impl.D)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc16_14.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc16_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %.loc16_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc16_14.3: Core.Form = init_form %.loc16_14.2 [concrete = constants.%.262]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .B = %T.as.I.impl.B.decl
// CHECK:STDOUT:     .C = %T.as.I.impl.C.decl
// CHECK:STDOUT:     .D = %T.as.I.impl.D.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.A(@T.as.I.impl.812d8b.1.%T.loc4_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.B(@T.as.I.impl.812d8b.2.%T.loc11_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.C(@T.as.I.impl.812d8b.2.%T.loc11_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_26.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_27: init %empty_tuple.type = converted %.loc15_26.1, %.loc15_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc15_27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.D(@T.as.I.impl.812d8b.2.%T.loc11_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.812d8b.2(%T) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.C.specific_fn.loc17_12.2: <specific function> = specific_function %T.as.I.impl.C, @T.as.I.impl.C(%T) [symbolic = %T.as.I.impl.C.specific_fn.loc17_12.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = specific_constant @T.as.I.impl.812d8b.2.%T.as.I.impl.C.decl, @T.as.I.impl.812d8b.2(constants.%T) [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %C.ref: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = name_ref C, %.loc17 [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %T.as.I.impl.C.specific_fn.loc17_12.1: <specific function> = specific_function %C.ref, @T.as.I.impl.C(constants.%T) [symbolic = %T.as.I.impl.C.specific_fn.loc17_12.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:     %T.as.I.impl.C.call: init %empty_tuple.type = call %T.as.I.impl.C.specific_fn.loc17_12.1()
// CHECK:STDOUT:     return %T.as.I.impl.C.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.812d8b.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT:   %.loc4_30 => constants.%.917
// CHECK:STDOUT:   %I.impl_witness.loc4_30.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type => constants.%T.as.I.impl.A.type
// CHECK:STDOUT:   %T.as.I.impl.A => constants.%T.as.I.impl.A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.812d8b.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc11_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_15.1 => constants.%T
// CHECK:STDOUT:   %.loc11_30 => constants.%.917
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.B.type => constants.%T.as.I.impl.B.type
// CHECK:STDOUT:   %T.as.I.impl.B => constants.%T.as.I.impl.B
// CHECK:STDOUT:   %T.as.I.impl.C.type => constants.%T.as.I.impl.C.type
// CHECK:STDOUT:   %T.as.I.impl.C => constants.%T.as.I.impl.C
// CHECK:STDOUT:   %T.as.I.impl.D.type => constants.%T.as.I.impl.D.type
// CHECK:STDOUT:   %T.as.I.impl.D => constants.%T.as.I.impl.D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.B(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.D(constants.%T) {}
// CHECK:STDOUT:
